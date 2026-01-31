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

# 类别映射
CLASS_MAP = {0: 'Trash', 1: 'Soft', 2: 'Perfect'}
# ============================================

class XMPManager:
    """XMP 元数据管理器 (中文 Lightroom 适配版)"""
    
    @staticmethod
    def generate_xmp_content(rating=0, label=""):
        """
        生成 XMP 内容
        注意：针对中文版 Lightroom，Label 必须写中文 "红色", "绿色" 等
        """
        # 使用 rdf:Description 属性方式写入，这是最古老但也最稳健的格式
        # 移除了 crs:Pick，因为它不稳定。改用颜色和星星。
        xmp_template = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="{rating}"
    xmp:Label="{label}">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
        return xmp_template

    @staticmethod
    def write_tag(image_path, category_idx):
        """写入 XMP 标记 (使用中文颜色)"""
        xmp_path = os.path.splitext(image_path)[0] + ".xmp"
        
        # === 核心修改：中文颜色映射 ===
        if category_idx == 0:   # Trash -> 紫色 + 1星
            label_val = "紫色"  
            rating_val = 1
            
        elif category_idx == 1: # Soft -> 红色 + 3星
            label_val = "红色"
            rating_val = 3
            
        elif category_idx == 2: # Perfect -> 绿色 + 5星
            label_val = "绿色"
            rating_val = 5
        
        else: # Unidentified
            label_val = "黄色" # 黄色表示未识别/异常
            rating_val = 0

        xml_content = XMPManager.generate_xmp_content(rating=rating_val, label=label_val)
        
        try:
            with open(xmp_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            return True
        except Exception as e:
            print(f"写入 XMP 失败: {e}")
            return False

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 RAW/JPG 的文件夹")
    root.destroy()
    return path

# 升级版特征提取函数 (Mask-Aware V2)
def extract_features_for_prediction(roi_img):
    if roi_img is None: return None
    if len(roi_img.shape) == 3: gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else: gray = roi_img

    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    h, w = gray.shape
    erode_size = max(3, int(min(h, w) * 0.01))
    kernel = np.ones((erode_size, erode_size), np.uint8)
    inner_mask = cv2.erode(binary_mask, kernel, iterations=2)
    
    valid_pixels = cv2.countNonZero(inner_mask)
    if valid_pixels == 0:
        inner_mask = binary_mask
        if cv2.countNonZero(inner_mask) == 0: return [0]*6

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    valid_gradients = magnitude[inner_mask > 0]
    
    if len(valid_gradients) == 0: return [0]*6

    feat_peak_grad = np.percentile(valid_gradients, 95)
    feat_mean_grad = np.mean(valid_gradients)
    feat_std_grad = np.std(valid_gradients)

    feat_fft_mid = calculate_sharpness_fft(roi_img, low_cut=0.10, high_cut=0.30)
    feat_fft_high = calculate_sharpness_fft(roi_img, low_cut=0.30, high_cut=0.70)
    
    valid_brightness = gray[inner_mask > 0]
    feat_bright = np.mean(valid_brightness) if len(valid_brightness) > 0 else 0

    return [feat_peak_grad, feat_mean_grad, feat_std_grad, feat_fft_mid, feat_fft_high, feat_bright]

def main():
    print("=== 自动鸟类筛选工具 (中文LR适配版) ===")
    
    if not os.path.exists(MODEL_FILE):
        print(f"错误: 未找到模型文件 {MODEL_FILE}")
        return
        
    try:
        clf = joblib.load(MODEL_FILE)
        print(f"✅ 已加载 AI 模型")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # === 用户交互选择模式 ===
    print("\n请选择工作模式:")
    print("  [1] 整理模式: 物理移动文件到文件夹")
    print("  [2] 标注模式: 生成 XMP (中文颜色标签，适配 Lightroom)")
    
    while True:
        mode = input("请输入 1 或 2: ").strip()
        if mode == '1':
            AUTO_MOVE_FILES = True
            AUTO_WRITE_XMP = False
            print(">> 已选择: 整理模式")
            break
        elif mode == '2':
            AUTO_MOVE_FILES = False
            AUTO_WRITE_XMP = True
            print(">> 已选择: 标注模式")
            break
        else:
            print("输入无效。")

    input_folder = select_folder()
    if not input_folder: return

    if AUTO_MOVE_FILES:
        for folder_name in CLASS_MAP.values():
            os.makedirs(os.path.join(input_folder, folder_name), exist_ok=True)
        os.makedirs(os.path.join(input_folder, "Unidentified"), exist_ok=True)

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_exts]
    
    detector = BirdDetector()
    results_stats = {'Trash': 0, 'Soft': 0, 'Perfect': 0, 'Unidentified': 0}

    print(f"\n开始处理 {len(file_list)} 张照片...")

    for filename in tqdm(file_list, desc="AI Sorting"):
        file_path = os.path.join(input_folder, filename)
        
        # 1. 加载与裁切
        full_img = load_best_available_image(file_path)
        if full_img is None: continue
        
        roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
        
        category_idx = -1 
        
        if roi is not None:
            feats = extract_features_for_prediction(roi)
            try:
                category_idx = clf.predict([feats])[0]
            except ValueError:
                pass
        
        label_name = CLASS_MAP[category_idx] if category_idx != -1 else "Unidentified"
        results_stats[label_name] += 1

        # 执行 XMP 标注
        if AUTO_WRITE_XMP:
            # 只有识别出类别的才标颜色，未识别的标黄色/0星
            XMPManager.write_tag(file_path, category_idx)

        # 执行 物理移动
        if AUTO_MOVE_FILES:
            target_folder = label_name
            try:
                dest_path = os.path.join(input_folder, target_folder, filename)
                shutil.move(file_path, dest_path)
                xmp_file = os.path.splitext(file_path)[0] + ".xmp"
                if os.path.exists(xmp_file):
                    shutil.move(xmp_file, os.path.join(input_folder, target_folder, os.path.basename(xmp_file)))
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("处理完成！统计结果：")
    print(f"🟣 Trash (紫色/1星) : {results_stats['Trash']}")
    print(f"🔴 Soft  (红色/3星) : {results_stats['Soft']}")
    print(f"🟢 Perfect(绿色/5星): {results_stats['Perfect']}")
    print("-" * 60)
    
    if AUTO_WRITE_XMP:
        print("✅ XMP 已生成 (中文标签)。")
        print("📥 Lightroom 使用技巧:")
        print("   1. 导入照片前，确保 Lightroom 界面语言为【中文】。")
        print("   2. 如果已导入，选中照片 -> 右键 -> 元数据 -> 从文件读取元数据。")
    print("=" * 60)

if __name__ == "__main__":
    main()