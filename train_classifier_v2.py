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

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 引入核心
from detect_birds_multi import BirdDetector, load_best_available_image
from metadata_utils import get_iso_speed

# ================= 极速配置 =================
ROI_STANDARD_SIZE = 1600
BATCH_SIZE = 16       # 每次并行处理多少张图片 (根据显存调整)
NUM_WORKERS = 8       # IO 线程数
CLASS_NAMES = ['Trash', 'Soft', 'Perfect']
# ===========================================

class GPUBatchFeatureExtractor:
    """
    [核心引擎] GPU 批量特征提取器 (Mask-Aware V2)
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ 特征提取引擎: {self.device.upper()} (Batch Mode)")
        
        # 预加载卷积核到 GPU
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
                    # BGR -> Gray
                    tensor = 0.114 * tensor[:,:,0] + 0.587 * tensor[:,:,1] + 0.299 * tensor[:,:,2]
                
                h, w = tensor.shape
                # Zero-pad 到标准尺寸
                pad_h, pad_w = target_h - h, target_w - w
                tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), "constant", 0)
            
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return None
        return torch.stack(batch_tensors).to(self.device)

    def extract_batch(self, roi_list):
        """批量提取 6 维图像特征"""
        if not roi_list: return []
        
        input_tensor = self._prepare_tensor_batch(roi_list)
        if input_tensor is None: return []
        
        batch_size = input_tensor.shape[0]
        
        # 1. 生成内部掩膜 (Mask-Aware)
        mask_binary = (input_tensor > 1.0).float()
        erode_ksize = int(ROI_STANDARD_SIZE * 0.01) | 1
        pad = erode_ksize // 2
        # GPU 腐蚀
        inner_mask = -F.max_pool2d(-mask_binary, kernel_size=erode_ksize, stride=1, padding=pad)
        
        # 2. 梯度计算
        gx = F.conv2d(input_tensor, self.sobel_x, padding=1)
        gy = F.conv2d(input_tensor, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # 3. FFT 计算 (增加 CPU 回退机制以解决 CUFFT_INTERNAL_ERROR)
        try:
            fft_res = torch.fft.fft2(input_tensor)
            fft_shift = torch.fft.fftshift(fft_res)
            mag_spec = 20 * torch.log(torch.abs(fft_shift) + 1)
        except RuntimeError as e:
            if "CUFFT" in str(e) or "fft" in str(e):
                # 如果 GPU FFT 失败，回退到 CPU 计算
                # print("⚠️ CUFFT Error detected, falling back to CPU for FFT batch...")
                input_cpu = input_tensor.cpu()
                fft_res_cpu = torch.fft.fft2(input_cpu)
                fft_shift_cpu = torch.fft.fftshift(fft_res_cpu)
                mag_spec_cpu = 20 * torch.log(torch.abs(fft_shift_cpu) + 1)
                # 计算完送回 GPU 以便后续处理
                mag_spec = mag_spec_cpu.to(self.device)
            else:
                raise e
        
        # 4. 统计特征
        features_batch = []
        
        # 预计算频率掩膜坐标
        H, W = ROI_STANDARD_SIZE, ROI_STANDARD_SIZE
        if (H, W) not in self.freq_masks_cache:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=self.device).view(-1, 1) - cy
            x = torch.arange(W, device=self.device).view(1, -1) - cx
            dist_sq = x**2 + y**2
            max_r_sq = min(cy, cx)**2
            self.freq_masks_cache[(H, W)] = (dist_sq, max_r_sq)
            
        dist_sq, max_r_sq = self.freq_masks_cache[(H, W)]
        
        for i in range(batch_size):
            if roi_list[i] is None:
                features_batch.append([0]*6)
                continue
                
            curr_mag = magnitude[i, 0]
            curr_mask = inner_mask[i, 0]
            curr_spec = mag_spec[i, 0]
            
            # 只取 Mask 内部有效梯度
            valid_grads = curr_mag[curr_mask > 0.5]
            
            if valid_grads.numel() == 0:
                features_batch.append([0]*6)
                continue
            
            feat_peak = torch.quantile(valid_grads, 0.95).item()
            feat_mean = torch.mean(valid_grads).item()
            feat_std = torch.std(valid_grads).item()
            
            # 频域能量
            def get_energy(low_p, high_p):
                r2_low = max_r_sq * (low_p**2)
                r2_high = max_r_sq * (high_p**2)
                mask_band = (dist_sq >= r2_low) & (dist_sq <= r2_high)
                vals = curr_spec[mask_band]
                return torch.mean(vals).item() if vals.numel() > 0 else 0.0

            feat_fft_mid = get_energy(0.10, 0.30)
            feat_fft_high = get_energy(0.30, 0.70)
            
            # 亮度
            valid_pix = input_tensor[i, 0][curr_mask > 0.5]
            feat_bright = torch.mean(valid_pix).item() if valid_pix.numel() > 0 else 0
            
            features_batch.append([feat_peak, feat_mean, feat_std, feat_fft_mid, feat_fft_high, feat_bright])
            
        return features_batch

def process_image_job(args):
    """
    工作线程任务：
    1. 读取图片 (IO)
    2. 读取 ISO
    3. 识别所有鸟并裁切
    返回: (filename, iso, [roi_list])
    """
    path, detector = args
    try:
        # A. 读图
        full_img = load_best_available_image(path)
        if full_img is None: return None
        
        # B. 读 ISO
        iso = get_iso_speed(path)
        
        # C. 识别多鸟
        # detect_and_crop_all 内部会处理 mask 背景消除
        rois = detector.detect_and_crop_all(full_img, standard_size=ROI_STANDARD_SIZE)
        
        return (os.path.basename(path), iso, rois)
    except:
        return None

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择包含 labels_multi.csv 的文件夹")
    root.destroy()
    return path

def main():
    print("=== AI 模型训练 (V4.0 GPU极速版) ===")
    
    folder = select_folder()
    if not folder: return
    
    csv_file = os.path.join(folder, "labels_multi.csv")
    if not os.path.exists(csv_file):
        print("错误: 未找到 labels_multi.csv")
        return

    # 1. 解析 CSV 映射关系
    # labels_map: { "filename": { 0: label_val, 2: label_val } }
    # 这样我们可以知道哪张图的哪只鸟被标记了
    print("正在解析标记数据...")
    labels_map = {} 
    valid_files = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and len(row) >= 3:
                fname = row[0]
                try:
                    b_idx = int(row[1])
                    label = int(row[2])
                    if fname not in labels_map: labels_map[fname] = {}
                    labels_map[fname][b_idx] = label
                    valid_files.add(fname)
                except: continue

    file_list = sorted(list(valid_files))
    print(f"需处理图片: {len(file_list)} 张 (包含多个标记目标)")

    detector = BirdDetector()
    gpu_extractor = GPUBatchFeatureExtractor()
    
    X = []
    y = []
    
    # 进度条管理
    pbar = tqdm(total=len(file_list), desc="GPU Processing")
    
    # 2. 批量处理循环 (核心加速逻辑)
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(folder, f) for f in batch_files]
        
        # A. 并行 IO & 检测 (CPU 多线程)
        results_list = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_image_job, (p, detector)) for p in batch_paths]
            for f in futures:
                res = f.result()
                if res: results_list.append(res)
        
        # B. 整理 Batch 数据
        # 我们需要把这批图片里的所有"有效且已标记"的 ROI 收集起来发给 GPU
        rois_to_process = []
        metadata_to_process = [] # (iso, label)
        
        for fname, iso, rois in results_list:
            if fname not in labels_map: continue
            
            bird_dict = labels_map[fname]
            
            # 遍历这张图里找到的所有鸟
            for idx, roi in enumerate(rois):
                # 只有当这个 index 在 CSV 里有记录时，才放入训练集
                if idx in bird_dict:
                    label = bird_dict[idx]
                    
                    # 标签映射
                    if label == 1: cls = 0   # Trash
                    elif label == 2: cls = 1 # Soft
                    elif label >= 3: cls = 2 # Perfect
                    else: continue
                    
                    rois_to_process.append(roi)
                    metadata_to_process.append((iso, cls))
        
        # C. GPU 批量特征提取 (一次性算完)
        if rois_to_process:
            # 得到 [[peak, mean, ...], ...]
            img_feats_batch = gpu_extractor.extract_batch(rois_to_process)
            
            # D. 组合最终特征 (Img Feats + ISO)
            for j, img_feats in enumerate(img_feats_batch):
                iso, cls_label = metadata_to_process[j]
                
                # ISO 对数归一化
                feat_iso = np.log10(max(iso, 50))
                
                # 拼接: [6维图像特征] + [ISO] = 7维
                full_feats = img_feats + [feat_iso]
                
                X.append(full_feats)
                y.append(cls_label)
        
        pbar.update(len(batch_files))
        
    pbar.close()
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n样本库构建完成: {len(X)} 个有效样本")
    print(f"Trash: {np.sum(y==0)} | Soft: {np.sum(y==1)} | Perfect: {np.sum(y==2)}")
    
    if len(X) < 10:
        print("样本太少，无法训练。")
        return

    # 3. 训练模型 (随机森林)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n正在训练随机森林 (Grid Search)...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # 简单网格搜索
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 60)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"🏆 测试集准确率: {acc:.4f} | F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 特征重要性
    print("\n📊 特征重要性:")
    names = ["Peak", "Mean", "Std", "FFT_Mid", "FFT_High", "Bright", "ISO"]
    imps = best_clf.feature_importances_
    for i in np.argsort(imps)[::-1]:
        print(f"   {names[i]:<10}: {imps[i]:.4f}")

    joblib.dump(best_clf, "best_bird_model_v2.pkl")
    print("\n✅ 模型已保存: best_bird_model_v2.pkl")

if __name__ == "__main__":
    # Windows多进程保护
    import multiprocessing
    multiprocessing.freeze_support()
    main()