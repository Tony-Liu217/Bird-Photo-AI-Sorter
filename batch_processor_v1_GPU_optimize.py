import os
import shutil
import joblib
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# 引入核心模块
from detect_birds_single_maskenabled import BirdDetector, load_best_available_image

# ================= 极速版配置 =================
ROI_STANDARD_SIZE = 1600
MODEL_FILE = "best_bird_model_multiclass.pkl"
CLASS_MAP = {0: 'Trash', 1: 'Soft', 2: 'Perfect'}

# 批量大小 (根据显存调整, 16-32 比较安全)
BATCH_SIZE = 16 
# CPU 读取线程数
NUM_WORKERS = 8

# 开关
AUTO_WRITE_XMP = True 
AUTO_MOVE_FILES = False 
# ============================================

class GPUBatchFeatureExtractor:
    """
    [核心组件] GPU 批量特征提取器
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
        
        input_tensor = self._prepare_tensor_batch(roi_list)
        
        mask_binary = (input_tensor > 1.0).float()
        erode_ksize = int(ROI_STANDARD_SIZE * 0.01) | 1
        pad = erode_ksize // 2
        inner_mask = -F.max_pool2d(-mask_binary, kernel_size=erode_ksize, stride=1, padding=pad)
        
        gx = F.conv2d(input_tensor, self.sobel_x, padding=1)
        gy = F.conv2d(input_tensor, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        fft_res = torch.fft.fft2(input_tensor)
        fft_shift = torch.fft.fftshift(fft_res)
        mag_spec = 20 * torch.log(torch.abs(fft_shift) + 1)
        
        features_batch = []
        
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
                if vals.numel() == 0: return 0.0
                return torch.mean(vals).item()

            feat_fft_mid = get_energy(0.10, 0.30)
            feat_fft_high = get_energy(0.30, 0.70)
            
            valid_pix = input_tensor[i, 0][curr_mask > 0.5]
            feat_bright = torch.mean(valid_pix).item() if valid_pix.numel() > 0 else 0
            
            features_batch.append([feat_peak, feat_mean, feat_std, feat_fft_mid, feat_fft_high, feat_bright])
            
        return features_batch

class XMPManager:
    """XMP 元数据管理器"""
    @staticmethod
    def generate_xmp_content(rating=0, label=""):
        return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}" xmp:Label="{label}"></rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""

    @staticmethod
    def write_tag(image_path, category_idx):
        xmp_path = os.path.splitext(image_path)[0] + ".xmp"
        if category_idx == 0: label, rating = "紫色", 1
        elif category_idx == 1: label, rating = "红色", 3
        elif category_idx == 2: label, rating = "绿色", 5
        else: label, rating = "黄色", 0
        
        try:
            with open(xmp_path, 'w', encoding='utf-8') as f:
                f.write(XMPManager.generate_xmp_content(rating, label))
        except: pass

def process_single_image_cpu_part(args):
    """CPU 任务：读图 + 裁切"""
    file_path, detector = args
    try:
        full_img = load_best_available_image(file_path)
        if full_img is None: return None
        roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
        return roi
    except Exception:
        return None

def select_folder():
    """
    弹出文件夹选择框 (修复版)
    """
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    
    # 【关键修复】强制窗口置顶，防止被其他窗口遮挡
    root.attributes('-topmost', True)
    root.update()
    
    path = filedialog.askdirectory(title="选择包含 RAW/JPG 的文件夹")
    root.destroy()
    return path

def main():
    print("=== 自动鸟类筛选工具 (GPU 批量极速版) ===")
    
    # 1. 优先让用户选择文件夹 (提升交互体验，防止加载模型时用户以为卡死)
    input_folder = select_folder()
    if not input_folder:
        print("未选择文件夹，程序退出。")
        return

    # 2. 模式选择
    global AUTO_MOVE_FILES, AUTO_WRITE_XMP
    print("\n请选择工作模式:")
    print("  [1] 整理模式 (移动文件)")
    print("  [2] 标注模式 (生成 XMP)")
    while True:
        mode = input("请输入 1 或 2: ").strip()
        if mode == '1': AUTO_MOVE_FILES = True; AUTO_WRITE_XMP = False; break
        elif mode == '2': AUTO_MOVE_FILES = False; AUTO_WRITE_XMP = True; break
        else: print("无效输入，请重试。")

    if not os.path.exists(MODEL_FILE):
        print(f"错误: 未找到模型 {MODEL_FILE}")
        return
    
    print("\n正在加载 AI 模型 (可能需要几秒钟)...")
    clf = joblib.load(MODEL_FILE)
    print(f"✅ 模型加载完成")

    gpu_extractor = GPUBatchFeatureExtractor()

    if AUTO_MOVE_FILES:
        for folder_name in CLASS_MAP.values():
            os.makedirs(os.path.join(input_folder, folder_name), exist_ok=True)
        os.makedirs(os.path.join(input_folder, "Unidentified"), exist_ok=True)

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_exts]
    file_list.sort()
    
    print(f"\n开始处理 {len(file_list)} 张照片...")
    print(f"配置: Batch Size = {BATCH_SIZE}, Loading Threads = {NUM_WORKERS}")

    detector = BirdDetector()
    results_stats = {'Trash': 0, 'Soft': 0, 'Perfect': 0, 'Unidentified': 0}

    # === 批量处理主循环 ===
    for i in tqdm(range(0, len(file_list), BATCH_SIZE), desc="Batch Processing"):
        batch_files = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(input_folder, f) for f in batch_files]
        
        # 并行加载
        rois = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_single_image_cpu_part, (path, detector)) for path in batch_paths]
            rois = [f.result() for f in futures]

        # 批量特征
        batch_feats = gpu_extractor.extract_batch(rois)
        
        # 预测
        for idx, feats in enumerate(batch_feats):
            fname = batch_files[idx]
            fpath = batch_paths[idx]
            category_idx = -1
            
            if feats is not None:
                try:
                    category_idx = clf.predict([feats])[0]
                except: pass
            
            label_name = CLASS_MAP[category_idx] if category_idx != -1 else "Unidentified"
            results_stats[label_name] += 1
            
            # 操作
            if AUTO_WRITE_XMP and category_idx != -1:
                XMPManager.write_tag(fpath, category_idx)
            
            if AUTO_MOVE_FILES:
                try:
                    dest = os.path.join(input_folder, label_name, fname)
                    shutil.move(fpath, dest)
                    xmp = os.path.splitext(fpath)[0] + ".xmp"
                    if os.path.exists(xmp):
                        shutil.move(xmp, os.path.join(input_folder, label_name, os.path.basename(xmp)))
                except: pass

    print("\n" + "=" * 60)
    print("处理完成！统计结果：")
    print(f"🟣 Trash   : {results_stats['Trash']}")
    print(f"🔴 Soft    : {results_stats['Soft']}")
    print(f"🟢 Perfect : {results_stats['Perfect']}")
    print("=" * 60)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()