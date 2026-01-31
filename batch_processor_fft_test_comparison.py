import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import itertools

# 引入核心模块
from detect_birds_multi_maskenabled import BirdDetector, load_best_available_image
from sharpness_evaluator_by_frequence import calculate_sharpness_fft

# ================= 配置区域 =================
ROI_STANDARD_SIZE = 1600 

# 开启自动寻优
ENABLE_AUTO_TUNING = True

# 默认兜底参数
FIXED_LOW_CUT = 0.05
FIXED_HIGH_CUT = 0.65 
FIXED_SIGMA = 0.5
FIXED_EXPONENT = 2.0

# 升级后的搜索空间 (加入 exponent)
SEARCH_RANGES = {
    # Low Cut: 0.05 之前表现最好，我们微调附近
    'low_cut': [0.05],
    
    # High Cut: 之前 0.65 表现最好，尝试更高
    'high_cut': [0.65],
    
    # Sigma: 中心加权
    'sigma': [0.65],
    
    # [新] 幂律指数: 关键变量！
    # 测试从线性(1.0)到高阶(3.5)的各种可能性
    'exponent': [1.5, 2.0, 2.5, 3.0, 3.5]
}
# ============================================

def get_gaussian_mask(shape, sigma=0.5):
    rows, cols = shape[:2]
    center_y, center_x = rows / 2, cols / 2
    y, x = np.ogrid[:rows, :cols]
    if rows == 0 or cols == 0: return np.ones(shape)
    y_norm = (y - center_y) / (rows / 2)
    x_norm = (x - center_x) / (cols / 2)
    mask = np.exp(-(x_norm**2 + y_norm**2) / (2 * sigma**2))
    return mask

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 labels.csv 的文件夹")
    root.destroy()
    return path

def calculate_score_for_roi(roi, low_cut, high_cut, sigma, exponent):
    """
    计算函数：支持动态指数
    Score = (FFT_Log_Val ^ exponent) / scaling_factor
    """
    if roi is None: return 0.0
    
    if len(roi.shape) == 3: gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else: gray = roi
        
    mask = get_gaussian_mask(gray.shape, sigma=sigma)
    weighted_roi = (gray.astype(float) * mask).astype(np.uint8)

    # 获取 FFT 对数能量均值 (通常在 100-250 之间)
    raw_val = calculate_sharpness_fft(weighted_roi, low_cut=low_cut, high_cut=high_cut)
    
    # 应用幂律变换
    # 除以 100^(exponent-1) 是为了保持数值在一个可读的范围内(例如 0-1000)，不影响线性回归的相关性 R值
    scaling = 100.0 ** (exponent - 1)
    return (raw_val ** exponent) / scaling

def main():
    print("=== 算法线性回归验证工具 (V2.0 幂律寻优版) ===")
    
    folder = select_folder()
    if not folder: return
    
    csv_file = os.path.join(folder, "labels.csv")
    if not os.path.exists(csv_file):
        print("未找到 labels.csv")
        return

    # 1. 读取数据
    data_map = {} 
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and len(row) >= 2:
                try:
                    data_map[row[0]] = int(row[1])
                except ValueError:
                    continue

    print(f"载入 {len(data_map)} 个标记样本，正在预处理 ROI...")

    # 2. 缓存 ROI (内存加速)
    detector = BirdDetector()
    cached_data = [] 
    
    for filename, label in tqdm(data_map.items(), desc="Preprocessing"):
        path = os.path.join(folder, filename)
        full_img = load_best_available_image(path)
        if full_img is None: continue

        roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
        if roi is not None:
            cached_data.append({'label': label, 'roi': roi})

    if not cached_data: return

    # 3. 四维网格搜索
    best_params = {
        'low_cut': FIXED_LOW_CUT, 'high_cut': FIXED_HIGH_CUT, 
        'sigma': FIXED_SIGMA, 'exponent': FIXED_EXPONENT
    }

    if ENABLE_AUTO_TUNING:
        print("\n正在运行四维网格搜索 (Low / High / Sigma / Exponent)...")
        best_r2 = -float('inf')
        
        combinations = list(itertools.product(
            SEARCH_RANGES['low_cut'],
            SEARCH_RANGES['high_cut'],
            SEARCH_RANGES['sigma'],
            SEARCH_RANGES['exponent']
        ))
        
        y_true = [d['label'] for d in cached_data]
        
        for low, high, sigma, exp in tqdm(combinations, desc="Tuning"):
            if low >= high: continue
            
            # 批量计算
            y_scores = [calculate_score_for_roi(d['roi'], low, high, sigma, exp) for d in cached_data]
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_scores)
            r_sq = r_value ** 2
            
            if r_sq > best_r2:
                best_r2 = r_sq
                best_params = {'low_cut': low, 'high_cut': high, 'sigma': sigma, 'exponent': exp}
        
        print(f"\n🏆 最佳参数组合找到 (R² = {best_r2:.4f}):")
        print(f"   Low Cut  : {best_params['low_cut']}")
        print(f"   High Cut : {best_params['high_cut']}")
        print(f"   Sigma    : {best_params['sigma']}")
        print(f"   Exponent : {best_params['exponent']} (幂律指数)")
        print("-" * 60)

    # 4. 生成最终报告
    x_labels = []
    y_scores = []
    
    for d in cached_data:
        score = calculate_score_for_roi(
            d['roi'], 
            best_params['low_cut'], best_params['high_cut'], 
            best_params['sigma'], best_params['exponent']
        )
        x_labels.append(d['label'])
        y_scores.append(score)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_labels, y_scores)
    r_squared = r_value ** 2

    print(f"\n📊 最终回归报告")
    print(f"相关系数 (r): {r_value:.4f}")
    print(f"决定系数 (R²): {r_squared:.4f}")
    
    # 5. 可视化
    plt.figure(figsize=(10, 6))
    x_jitter = np.array(x_labels) + np.random.normal(0, 0.05, size=len(x_labels))
    plt.scatter(x_jitter, y_scores, alpha=0.6, edgecolors='w')

    x_fit = np.linspace(min(x_labels), max(x_labels), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r--', label=f'Fit: $R^2$={r_squared:.2f}')

    plt.title(f'Best Fit: Exponent = {best_params["exponent"]}')
    plt.xlabel('Human Label')
    plt.ylabel(f'Score (FFT ^ {best_params["exponent"]})')
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()