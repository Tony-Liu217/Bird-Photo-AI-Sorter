import os
import csv
import joblib
import numpy as np
# import pandas as pd  <-- 移除 pandas 依赖
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import cv2  # 确保导入 cv2
import matplotlib.pyplot as plt
from scipy import stats

# 引入核心
from detect_birds_multi_maskenabled import BirdDetector, load_best_available_image
from sharpness_evaluator_by_frequence import calculate_sharpness_fft

ROI_STANDARD_SIZE = 1600

# 定义类别名称映射 (用于显示)
CLASS_NAMES = ['Trash', 'Soft', 'Perfect'] # 对应 0, 1, 2

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 labels.csv 的文件夹")
    root.destroy()
    return path

# === 升级版特征提取器 (Mask-Aware V2) ===
def extract_enhanced_features(roi_img):
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
    print("=== AI 模型深度优化 (随机森林版) ===")
    print("类别定义: 0-Trash(废片), 1-Soft(部分失焦), 2-Perfect(完美)")
    
    folder = select_folder()
    if not folder: return
    
    csv_file = os.path.join(folder, "labels.csv")
    
    print("正在加载数据与提取增强特征 (Mask-Aware)...")
    data_files = []
    X = []
    y = []
    
    detector = BirdDetector()
    
    # 读取 CSV
    raw_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and len(row) >= 2:
                try:
                    raw_data.append((row[0], int(row[1])))
                except: continue

    # 提取特征
    for fname, label in tqdm(raw_data):
        path = os.path.join(folder, fname)
        img = load_best_available_image(path)
        if img is None: continue
        
        roi, _ = detector.detect_and_crop(img, standard_size=ROI_STANDARD_SIZE)
        if roi is None: continue
        
        feats = extract_enhanced_features(roi)
        
        # === 核心修改：三级分类映射 ===
        # 假设 CSV 中 1=Trash, 2=Soft, 3=Perfect (兼容旧数据的 4 为 Perfect)
        if label == 1:
            cls_label = 0 # Trash
        elif label == 2:
            cls_label = 1 # Soft
        elif label >= 3:
            cls_label = 2 # Perfect
        else:
            continue # 跳过异常标签
        
        X.append(feats)
        y.append(cls_label)
        data_files.append(fname)

    X = np.array(X)
    y = np.array(y)
    
    print(f"\n样本准备完毕: {len(X)} 张")
    print(f"分布: Trash={np.sum(y==0)}, Soft={np.sum(y==1)}, Perfect={np.sum(y==2)}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(data_files)), test_size=0.2, random_state=42
    )

    # === 随机森林深度调优 (Hyperparameter Tuning) ===
    print("\n正在进行随机森林参数网格搜索 (这可能需要几分钟)...")
    
    # 定义基础模型
    rf = RandomForestClassifier(random_state=42)
    
    # 定义参数网格
    # 包含树的数量、深度、以及分裂标准，寻找最佳平衡点
    param_grid = {
        'n_estimators': [100, 200, 300],        # 树的数量
        'max_depth': [None, 10, 20, 30],        # 树的深度 (防止过拟合)
        'min_samples_split': [2, 5, 10],        # 节点分裂所需最小样本数
        'min_samples_leaf': [1, 2, 4],          # 叶子节点最小样本数
        'max_features': ['sqrt', 'log2']
    }
    
    # 建立网格搜索 (使用 F1-macro 作为评分标准，兼顾废片和好片的平衡)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=5,           # 5折交叉验证
        n_jobs=-1,      # 使用所有CPU核心并行计算
        verbose=1,
        scoring='f1_macro'
    )
    
    grid_search.fit(X_train, y_train)
    
    print("-" * 60)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳训练集验证分数: {grid_search.best_score_:.4f}")
    
    # 获取最佳模型
    best_clf = grid_search.best_estimator_
    
    # 在独立的测试集上验证
    y_pred = best_clf.predict(X_test)
    
    # 计算最终指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 60)
    print(f"🏆 测试集评估结果:")
    print(f"   准确率 (Accuracy): {acc:.4f}")
    print(f"   综合分 (F1 Score): {f1:.4f}")
    
    # 打印详细三分类报告
    rpt = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(f"   [Trash] Recall:   {rpt['Trash']['recall']:.2f} (抓废片能力)")
    print(f"   [Soft]  F1-Score: {rpt['Soft']['f1-score']:.2f}")
    print(f"   [Perfect] Prec:   {rpt['Perfect']['precision']:.2f} (好片纯度)")
    
    # 保存最佳模型 (覆盖旧文件)
    joblib.dump(best_clf, "best_bird_model_multiclass.pkl")
    print("\n✅ 优化后的最佳模型已保存为 best_bird_model_multiclass.pkl")

    # === 错误分析 ===
    print("\n正在生成错误分析报告 (Error Analysis)...")
    
    final_pred = best_clf.predict(X_test)
    test_files = [data_files[i] for i in idx_test]
    
    errors = []
    for i in range(len(y_test)):
        if y_test[i] != final_pred[i]:
            true_str = CLASS_NAMES[y_test[i]]
            pred_str = CLASS_NAMES[final_pred[i]]
            
            # 判断错误类型严重程度
            err_type = "严重" if abs(y_test[i] - final_pred[i]) == 2 else "轻微"
            
            errors.append({
                'Filename': test_files[i],
                'True_Label': true_str,
                'AI_Predict': pred_str,
                'Severity': err_type
            })
    
    if errors:
        try:
            with open("error_analysis_v2.csv", 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['Filename', 'True_Label', 'AI_Predict', 'Severity'])
                writer.writeheader()
                writer.writerows(errors)
            print(f"⚠️ 发现了 {len(errors)} 个误判。详情已写入 error_analysis_v2.csv")
            print("建议重点检查 '严重' 类型的误判 (Trash vs Perfect)。")
        except Exception as e:
            print(f"无法保存错误分析报告: {e}")
    else:
        print("🎉 完美！测试集上没有误判。")

    # === 线性分析绘图报告 ===
    print("\n正在生成线性分析绘图报告...")
    
    # 1. 计算模型预测评分 (Weighted Score)
    # 将概率转换为连续分数: 0*P(Trash) + 1*P(Soft) + 2*P(Perfect)
    y_probs = best_clf.predict_proba(X_test)
    model_scores = y_probs[:, 0] * 0 + y_probs[:, 1] * 1 + y_probs[:, 2] * 2
    
    # 2. 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, model_scores)
    r_squared = r_value ** 2
    
    print("\n" + "=" * 60)
    print(f"📊 模型评分线性回归报告 (Test Set)")
    print("-" * 60)
    print(f"相关系数 (r)      : {r_value:.4f}")
    print(f"决定系数 (R²)     : {r_squared:.4f}")
    print(f"P值 (P-value)     : {p_value:.4e}")
    print("-" * 60)
    print(f"📈 回归方程: Predicted_Score = {slope:.2f} * True_Label + {intercept:.2f}")
    print("=" * 60)
    
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    # 添加抖动 (Jitter)
    jitter = np.random.normal(0, 0.05, size=len(y_test))
    plt.scatter(y_test + jitter, model_scores, alpha=0.6, c='blue', edgecolors='w', label='Test Samples')
    
    # 绘制回归线
    x_fit = np.linspace(min(y_test), max(y_test), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: $R^2$={r_squared:.2f}')
    
    # 理想线
    plt.plot([0, 2], [0, 2], 'g:', alpha=0.5, label='Ideal (y=x)')
    
    plt.title(f'Linear Regression: Model Score vs True Label')
    plt.xlabel('True Label (0:Trash, 1:Soft, 2:Perfect)')
    plt.ylabel('Model Weighted Score (0.0 - 2.0)')
    plt.xticks([0, 1, 2], ['Trash', 'Soft', 'Perfect'])
    plt.yticks(np.arange(0, 2.5, 0.5))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import cv2 
    main()