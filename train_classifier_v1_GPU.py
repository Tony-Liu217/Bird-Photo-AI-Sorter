import os
import csv
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# sklearn 机器学习库
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 引入核心模块
from detect_birds_single_maskenabled import BirdDetector, load_best_available_image

# ================= 配置区域 =================
ROI_STANDARD_SIZE = 1600
BATCH_SIZE = 16       # 显存允许的情况下越大越好
NUM_WORKERS = 8       # CPU 线程数
CLASS_NAMES = ['Trash', 'Soft', 'Perfect']
# ===========================================

class GPUBatchFeatureExtractor:
    """
    [核心组件] GPU 批量特征提取器
    完全复刻主程序逻辑，确保训练/推理一致性
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ 特征提取引擎: {self.device.upper()} (Batch Mode)")
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.freq_masks_cache = {}

    def _prepare_tensor_batch(self, roi_list):
        batch_tensors = []
        target_h, target_w = ROI_STANDARD_SIZE, ROI_STANDARD_SIZE
        
        for roi in roi_list:
            if roi is None:
                tensor = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            else:
                tensor = torch.from_numpy(roi).float()
                if len(tensor.shape) == 3:
                    tensor = 0.114 * tensor[:,:,0] + 0.587 * tensor[:,:,1] + 0.299 * tensor[:,:,2]
                
                h, w = tensor.shape
                pad_h, pad_w = target_h - h, target_w - w
                tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), "constant", 0)
            
            batch_tensors.append(tensor)
        return torch.stack(batch_tensors).to(self.device)

    def extract_batch(self, roi_list):
        if not roi_list: return []
        batch_size = len(roi_list)
        
        # 1. 预处理
        input_tensor = self._prepare_tensor_batch(roi_list)
        
        # 2. 掩膜与腐蚀
        mask_binary = (input_tensor > 1.0).float()
        erode_ksize = int(ROI_STANDARD_SIZE * 0.01) | 1
        pad = erode_ksize // 2
        inner_mask = -F.max_pool2d(-mask_binary, kernel_size=erode_ksize, stride=1, padding=pad)
        
        # 3. 梯度
        gx = F.conv2d(input_tensor, self.sobel_x, padding=1)
        gy = F.conv2d(input_tensor, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # 4. FFT
        fft_res = torch.fft.fft2(input_tensor)
        fft_shift = torch.fft.fftshift(fft_res)
        mag_spec = 20 * torch.log(torch.abs(fft_shift) + 1)
        
        features_batch = []
        
        # 预计算掩膜
        H, W = ROI_STANDARD_SIZE, ROI_STANDARD_SIZE
        if (H, W) not in self.freq_masks_cache:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=self.device).view(-1, 1) - cy
            x = torch.arange(W, device=self.device).view(1, -1) - cx
            dist_sq = x**2 + y**2
            max_r_sq = min(cy, cx)**2
            self.freq_masks_cache[(H, W)] = (dist_sq, max_r_sq)
            
        dist_sq, max_r_sq = self.freq_masks_cache[(H, W)]
        
        # 5. 提取统计值
        for i in range(batch_size):
            if roi_list[i] is None:
                features_batch.append(None)
                continue
                
            curr_mag = magnitude[i, 0]
            curr_mask = inner_mask[i, 0]
            curr_spec = mag_spec[i, 0]
            
            valid_grads = curr_mag[curr_mask > 0.5]
            if valid_grads.numel() == 0:
                features_batch.append([0]*6)
                continue
            
            feat_peak = torch.quantile(valid_grads, 0.95).item()
            feat_mean = torch.mean(valid_grads).item()
            feat_std = torch.std(valid_grads).item()
            
            def get_energy(low_p, high_p):
                r2_low = max_r_sq * (low_p**2)
                r2_high = max_r_sq * (high_p**2)
                mask_band = (dist_sq >= r2_low) & (dist_sq <= r2_high)
                vals = curr_spec[mask_band]
                return torch.mean(vals).item() if vals.numel() > 0 else 0.0

            feat_fft_mid = get_energy(0.10, 0.30)
            feat_fft_high = get_energy(0.30, 0.70)
            
            valid_pix = input_tensor[i, 0][curr_mask > 0.5]
            feat_bright = torch.mean(valid_pix).item() if valid_pix.numel() > 0 else 0
            
            features_batch.append([feat_peak, feat_mean, feat_std, feat_fft_mid, feat_fft_high, feat_bright])
            
        return features_batch

def process_single_image_cpu_part(args):
    """CPU 任务：读图 + 裁切"""
    file_path, detector = args
    try:
        full_img = load_best_available_image(file_path)
        if full_img is None: return None
        roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
        return roi
    except: return None

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 labels.csv 的文件夹")
    root.destroy()
    return path

def main():
    print("=== AI 模型训练工具 (GPU 批量极速版) ===")
    
    folder = select_folder()
    if not folder: return
    
    csv_file = os.path.join(folder, "labels.csv")
    if not os.path.exists(csv_file):
        print("未找到 labels.csv")
        return

    # 1. 读取 CSV 标签
    print("正在读取标签...")
    data_entries = [] # list of (filename, label_idx)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 2: continue
            try:
                raw_label = int(row[1])
                # 映射: 1->Trash(0), 2->Soft(1), 3->Perfect(2)
                if raw_label == 1: cls = 0
                elif raw_label == 2: cls = 1
                elif raw_label >= 3: cls = 2
                else: continue
                data_entries.append((row[0], cls))
            except: continue

    print(f"有效样本数: {len(data_entries)}")
    
    # 2. 批量特征提取
    detector = BirdDetector()
    gpu_extractor = GPUBatchFeatureExtractor()
    
    X = []
    y = []
    filenames = []
    
    # 进度条
    pbar = tqdm(total=len(data_entries), desc="GPU Feature Extraction")
    
    # 分批次处理
    for i in range(0, len(data_entries), BATCH_SIZE):
        batch_slice = data_entries[i : i + BATCH_SIZE]
        batch_files = [item[0] for item in batch_slice]
        batch_labels = [item[1] for item in batch_slice]
        batch_paths = [os.path.join(folder, f) for f in batch_files]
        
        # A. 并行 IO & Crop
        rois = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_single_image_cpu_part, (path, detector)) for path in batch_paths]
            rois = [f.result() for f in futures]
            
        # B. GPU 批量提取
        feats_list = gpu_extractor.extract_batch(rois)
        
        # C. 收集结果
        for j, feats in enumerate(feats_list):
            if feats is not None:
                X.append(feats)
                y.append(batch_labels[j])
                filenames.append(batch_files[j])
                
        pbar.update(len(batch_slice))
    
    pbar.close()
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n最终训练集: {len(X)} 张")
    print(f"Trash: {np.sum(y==0)}, Soft: {np.sum(y==1)}, Perfect: {np.sum(y==2)}")
    
    # 3. 训练模型 (随机森林网格搜索)
    print("\n正在训练模型 (Random Forest Grid Search)...")
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, range(len(filenames)), test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1,
        verbose=1,
        scoring='f1_macro'
    )
    
    grid_search.fit(X_train, y_train)
    
    print("-" * 60)
    print(f"最佳参数: {grid_search.best_params_}")
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"🏆 测试集准确率: {acc:.4f} | F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 保存
    model_path = "best_bird_model_multiclass.pkl"
    joblib.dump(best_clf, model_path)
    print(f"\n✅ 模型已保存为: {model_path}")
    
    # 4. 错误分析
    print("\n生成错误报告...")
    errors = []
    final_pred = best_clf.predict(X_test)
    test_files = [filenames[i] for i in idx_test]
    
    for i in range(len(y_test)):
        if y_test[i] != final_pred[i]:
            true_l = CLASS_NAMES[y_test[i]]
            pred_l = CLASS_NAMES[final_pred[i]]
            severity = "严重" if abs(y_test[i] - final_pred[i]) == 2 else "轻微"
            errors.append({
                'Filename': test_files[i],
                'True': true_l,
                'Predicted': pred_l,
                'Severity': severity
            })
            
    if errors:
        try:
            with open("error_analysis_gpu.csv", 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['Filename', 'True', 'Predicted', 'Severity'])
                writer.writeheader()
                writer.writerows(errors)
            print(f"⚠️ {len(errors)} 个误判已写入 error_analysis_gpu.csv")
        except: pass
    else:
        print("🎉 测试集无误判！")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()