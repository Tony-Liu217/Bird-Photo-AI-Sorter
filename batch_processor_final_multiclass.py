import os
import shutil
import joblib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# 引入核心模块
from detect_birds_multi_maskenabled import BirdDetector, load_best_available_image
from sharpness_evaluator_by_frequence import calculate_sharpness_fft

# ================= 生产配置 =================
ROI_STANDARD_SIZE = 1600
MODEL_FILE = "best_bird_model_multiclass.pkl"

# 类别映射 (必须与训练时一致)
CLASS_MAP = {0: 'Trash', 1: 'Soft', 2: 'Perfect'}

# 是否自动移动文件? (True: 移动到子文件夹; False: 仅生成报告)
AUTO_MOVE_FILES = True
# ============================================

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 RAW/JPG 的文件夹")
    root.destroy()
    return path

# 升级版特征提取函数 (Mask-Aware V2)
# 必须与 train_classifier_v2.py 中的 extract_enhanced_features 完全一致
def extract_features_for_prediction(roi_img):
    """
    针对【纯黑背景 ROI】优化的特征提取器
    核心逻辑：排除遮罩边缘干扰，只计算身体内部的锐度
    """
    if roi_img is None: return None
    
    # 1. 预处理
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # 2. 生成"内部掩膜" (Inner Mask)
    # 逻辑：找出非黑区域 -> 向内腐蚀 -> 只计算这个范围内的梯度
    # 这样能彻底避开抠图产生的锐利边缘
    
    # 二值化找出鸟的区域 (非0即为鸟)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 动态计算腐蚀量 (约占短边的 1%)
    h, w = gray.shape
    erode_size = max(3, int(min(h, w) * 0.01))
    kernel = np.ones((erode_size, erode_size), np.uint8)
    
    # 向内腐蚀，得到"绝对安全的内部区域"
    inner_mask = cv2.erode(binary_mask, kernel, iterations=2)
    
    # 计算有效像素数
    valid_pixels = cv2.countNonZero(inner_mask)
    if valid_pixels == 0:
        # 鸟太小，腐蚀没了，回退到原始 mask
        inner_mask = binary_mask
        valid_pixels = cv2.countNonZero(inner_mask)
        if valid_pixels == 0: return [0]*6 # 纯黑图

    # 3. 计算梯度 (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    
    # === 关键步骤：只提取 inner_mask 区域内的梯度值 ===
    valid_gradients = magnitude[inner_mask > 0]
    
    if len(valid_gradients) == 0: return [0]*6

    # 特征 A: 内部峰值梯度 (95% 分位) - 区分 Soft/Perfect 的核心
    feat_peak_grad = np.percentile(valid_gradients, 95)
    
    # 特征 B: 内部平均梯度 - 反映整体纹理
    feat_mean_grad = np.mean(valid_gradients)
    
    # 特征 C: 梯度标准差 - 反映纹理复杂度
    feat_std_grad = np.std(valid_gradients)

    # 4. FFT 频域分析 (辅助判断噪点)
    # FFT 难以完全排除边缘效应，但作为辅助特征依然有效
    feat_fft_mid = calculate_sharpness_fft(roi_img, low_cut=0.10, high_cut=0.30)
    feat_fft_high = calculate_sharpness_fft(roi_img, low_cut=0.30, high_cut=0.70)
    
    # 5. 内部亮度统计
    valid_brightness = gray[inner_mask > 0]
    feat_bright = np.mean(valid_brightness) if len(valid_brightness) > 0 else 0

    return [
        feat_peak_grad,   # 最重要：眼睛/羽毛有多锐
        feat_mean_grad,   # 整体质感
        feat_std_grad,    # 纹理丰富度
        feat_fft_mid,     # 频域中频
        feat_fft_high,    # 频域高频 (噪点检测)
        feat_bright       # 亮度 (防止过暗误判)
    ]

def main():
    print("=== 自动鸟类筛选与归档工具 (AI 最终版 V3) ===")
    print("特征策略: 内部梯度峰值 + 内部纹理 + FFT (排除边缘效应)")
    
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 未找到模型文件 {MODEL_FILE}")
        print("提示: 请确保您已使用新的特征提取逻辑重新训练了模型！")
        return
        
    try:
        clf = joblib.load(MODEL_FILE)
        print(f"✅ 已加载 AI 模型")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    input_folder = select_folder()
    if not input_folder: return

    # 准备子文件夹
    if AUTO_MOVE_FILES:
        for folder_name in CLASS_MAP.values():
            folder_path = os.path.join(input_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
        # 额外创建一个"未识别"文件夹
        os.makedirs(os.path.join(input_folder, "Unidentified"), exist_ok=True)

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_exts]
    
    detector = BirdDetector()
    results_stats = {'Trash': 0, 'Soft': 0, 'Perfect': 0, 'Unidentified': 0}

    print(f"开始处理 {len(file_list)} 张照片...")

    for filename in tqdm(file_list, desc="AI Sorting"):
        file_path = os.path.join(input_folder, filename)
        
        # 1. 加载与裁切
        full_img = load_best_available_image(file_path)
        if full_img is None: continue
        
        roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
        
        target_folder = "Unidentified"
        confidence = 0.0
        
        if roi is not None:
            # 2. 提取特征 (使用新的增强版函数)
            feats = extract_features_for_prediction(roi)
            
            try:
                # 3. AI 预测
                pred_idx = clf.predict([feats])[0]
                probs = clf.predict_proba([feats])[0]
                
                target_folder = CLASS_MAP[pred_idx]
                confidence = probs[pred_idx] * 100
            except ValueError:
                print(f"\n⚠️ 特征维度不匹配！请确保使用 train_classifier_v2.py 重新训练了模型。")
                print(f"当前提取特征数: {len(feats)}")
                return
        else:
            pass

        results_stats[target_folder] += 1

        # 4. 移动文件 (或重命名)
        if AUTO_MOVE_FILES:
            try:
                # 构造目标路径
                dest_path = os.path.join(input_folder, target_folder, filename)
                shutil.move(file_path, dest_path)
                
                # 可选：如果伴随有同名 XMP 文件，也一并移动
                xmp_file = os.path.splitext(file_path)[0] + ".xmp"
                if os.path.exists(xmp_file):
                    dest_xmp = os.path.join(input_folder, target_folder, os.path.basename(xmp_file))
                    shutil.move(xmp_file, dest_xmp)
                    
            except Exception as e:
                print(f"移动文件 {filename} 失败: {e}")

    # 5. 总结报告
    print("\n" + "=" * 60)
    print("处理完成！统计结果：")
    print(f"🗑️  Trash (废片)   : {results_stats['Trash']}")
    print(f"😐 Soft (部分失焦): {results_stats['Soft']}")
    print(f"🏆 Perfect (完美) : {results_stats['Perfect']}")
    print(f"❓ Unidentified   : {results_stats['Unidentified']}")
    print("-" * 60)
    if AUTO_MOVE_FILES:
        print(f"所有文件已自动归类到 {input_folder} 下的子文件夹中。")
    print("=" * 60)

if __name__ == "__main__":
    main()