import os
import shutil
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 引入项目依赖
# 确保文件夹里有 detect_birds_multi.py 和 metadata_utils.py
try:
    from detect_birds_multi import BirdDetector, load_best_available_image
except ImportError:
    # 兼容旧文件名
    from detect_birds_multi import BirdDetector, load_best_available_image

try:
    from metadata_utils import get_iso_speed
except ImportError:
    import exifread
    def get_iso_speed(file_path):
        """简易内联版 ISO 读取器"""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                keys = ['EXIF ISOSpeedRatings', 'Image ISOSpeedRatings', 'EXIF ISO']
                for k in keys:
                    if k in tags: return int(str(tags[k]))
            return 800
        except: return 800

# ================= 生产环境配置 =================
ROI_STANDARD_SIZE = 1600
BATCH_SIZE = 16          # 显存允许时可调大 (16-32)
NUM_WORKERS = 8          # CPU 读取线程数
MODEL_FILE = "best_bird_model_v2.pkl" # 必须是含 ISO 特征的新模型

# 类别映射
CLASS_MAP = {0: 'Trash', 1: 'Soft', 2: 'Perfect'}

# 全局标志位 (将在运行时由用户选择决定)
AUTO_WRITE_XMP = False
AUTO_MOVE_FILES = False
# ===============================================

class XMPManager:
    """Lightroom 元数据生成器 (中文适配版)"""
    @staticmethod
    def generate_xmp_content(rating=0, label=""):
        return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="{rating}"
    xmp:Label="{label}">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""

    @staticmethod
    def write_tag(image_path, category_idx):
        xmp_path = os.path.splitext(image_path)[0] + ".xmp"
        
        # 中文 Lightroom 颜色映射
        if category_idx == 0:   # Trash
            label, rating = "紫色", 1 
        elif category_idx == 1: # Soft
            label, rating = "红色", 3
        elif category_idx == 2: # Perfect
            label, rating = "绿色", 5
        else:
            label, rating = "黄色", 0 # 未识别

        try:
            with open(xmp_path, 'w', encoding='utf-8') as f:
                f.write(XMPManager.generate_xmp_content(rating, label))
        except Exception as e:
            print(f"XMP写入失败: {e}")

class GPUBatchFeatureExtractor:
    """[核心组件] GPU 批量特征提取器 (含 FFT 异常回退机制)"""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ 推理引擎初始化: {self.device.upper()}")
        
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
                pad_h, pad_w = target_h - h, target_w - w
                tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), "constant", 0)
            
            batch_tensors.append(tensor)
        
        if not batch_tensors: return None
        return torch.stack(batch_tensors).to(self.device)

    def extract_batch(self, roi_list):
        if not roi_list: return []
        input_tensor = self._prepare_tensor_batch(roi_list)
        if input_tensor is None: return []
        
        batch_size = input_tensor.shape[0]
        
        # 1. Mask Aware Corresion
        mask_binary = (input_tensor > 1.0).float()
        erode_ksize = int(ROI_STANDARD_SIZE * 0.01) | 1
        pad = erode_ksize // 2
        inner_mask = -F.max_pool2d(-mask_binary, kernel_size=erode_ksize, stride=1, padding=pad)
        
        # 2. Gradient
        gx = F.conv2d(input_tensor, self.sobel_x, padding=1)
        gy = F.conv2d(input_tensor, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # 3. FFT (带 CPU Fallback)
        try:
            fft_res = torch.fft.fft2(input_tensor)
            fft_shift = torch.fft.fftshift(fft_res)
            mag_spec = 20 * torch.log(torch.abs(fft_shift) + 1)
        except RuntimeError as e:
            if "CUFFT" in str(e) or "fft" in str(e):
                input_cpu = input_tensor.cpu()
                fft_res = torch.fft.fft2(input_cpu)
                fft_shift = torch.fft.fftshift(fft_res)
                mag_spec = (20 * torch.log(torch.abs(fft_shift) + 1)).to(self.device)
            else:
                raise e
        
        features_batch = []
        
        # Frequency Masks Cache
        H, W = ROI_STANDARD_SIZE, ROI_STANDARD_SIZE
        if (H, W) not in self.freq_masks_cache:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=self.device).view(-1, 1) - cy
            x = torch.arange(W, device=self.device).view(1, -1) - cx
            dist_sq = x**2 + y**2
            max_r_sq = min(cy, cx)**2
            self.freq_masks_cache[(H, W)] = (dist_sq, max_r_sq)
        
        dist_sq, max_r_sq = self.freq_masks_cache[(H, W)]
        
        # 4. Statistics
        for i in range(batch_size):
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

def process_image_job(args):
    """IO 线程任务: 读图 + 读ISO + YOLO检测"""
    path, detector = args
    try:
        # A. 快速读取
        full_img = load_best_available_image(path)
        if full_img is None: return None
        
        # B. 读取 ISO
        iso = get_iso_speed(path)
        
        # C. YOLO 多目标检测
        rois = detector.detect_and_crop_all(full_img, standard_size=ROI_STANDARD_SIZE)
        
        if not rois: return None
        
        return (path, iso, rois)
    except Exception as e:
        return None

def select_folder():
    root = tk.Tk(); root.withdraw()
    # 强制置顶，防止窗口被遮挡
    root.attributes('-topmost', True)
    root.update()
    path = filedialog.askdirectory(title="选择要筛选的照片文件夹")
    root.destroy()
    return path

def main():
    print("=== 🦅 Bird Photo AI Sorter V2 (GPU Batch Dual Mode) ===")
    
    # 1. 模式选择 (放在最前面，防止用户等待模型加载时以为卡死)
    global AUTO_MOVE_FILES, AUTO_WRITE_XMP
    print("\n请选择工作模式:")
    print("  [1] 整理模式: 物理移动文件到 Trash/Soft/Perfect 文件夹")
    print("  [2] 标注模式: 生成 .xmp 文件 (推荐 Lightroom 用户)")
    
    while True:
        mode = input("请输入 1 或 2: ").strip()
        if mode == '1':
            AUTO_MOVE_FILES = True; AUTO_WRITE_XMP = False
            print(">> 已选择: 整理模式 (移动文件)")
            break
        elif mode == '2':
            AUTO_MOVE_FILES = False; AUTO_WRITE_XMP = True
            print(">> 已选择: 标注模式 (生成 XMP)")
            break
        else:
            print("输入无效。")

    # 2. 加载模型
    if not os.path.exists(MODEL_FILE):
        print(f"❌ 错误: 找不到模型文件 {MODEL_FILE}")
        print("请先运行 train_classifier_v2_multi_gpu.py 训练新模型")
        return
        
    print(f"\n正在加载模型 {MODEL_FILE}...")
    try:
        clf = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 选择文件夹
    input_folder = select_folder()
    if not input_folder: return

    # 4. 准备目标文件夹
    if AUTO_MOVE_FILES:
        for folder_name in CLASS_MAP.values():
            os.makedirs(os.path.join(input_folder, folder_name), exist_ok=True)
        os.makedirs(os.path.join(input_folder, "Unidentified"), exist_ok=True)

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.jpg', '.jpeg', '.png', '.tif'}
    file_list = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_exts]
    file_list.sort()
    
    print(f"📂 发现 {len(file_list)} 张图片，准备处理...")

    # 5. 初始化引擎
    detector = BirdDetector()
    gpu_extractor = GPUBatchFeatureExtractor()
    
    stats = {0: 0, 1: 0, 2: 0} # Trash, Soft, Perfect
    
    # 6. 批处理主循环
    pbar = tqdm(total=len(file_list), desc="🚀 AI Sorting")
    
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(input_folder, f) for f in batch_files]
        
        # --- Stage 1: 并行 IO & 检测 (CPU) ---
        job_results = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_image_job, (p, detector)) for p in batch_paths]
            for f in futures:
                res = f.result()
                if res: job_results.append(res)
        
        if not job_results:
            pbar.update(len(batch_files))
            continue
            
        # --- Stage 2: 准备 GPU Batch 数据 ---
        flat_rois = []
        map_info = [] 
        
        for job_idx, (path, iso, rois) in enumerate(job_results):
            for roi in rois:
                flat_rois.append(roi)
                map_info.append((job_idx, iso))
        
        # --- Stage 3: GPU 极速特征提取 ---
        if flat_rois:
            # 得到视觉特征 (6维)
            visual_feats_batch = gpu_extractor.extract_batch(flat_rois)
            
            # --- Stage 4: 组合特征 & 评分 ---
            # image_scores[job_idx] = [(prob_trash, prob_soft, prob_perfect), ...]
            image_scores = {}
            
            for k, visual_feats in enumerate(visual_feats_batch):
                job_idx, iso = map_info[k]
                
                # 拼接 ISO 特征 (第7维)
                feat_iso = np.log10(max(iso, 50))
                full_feats = visual_feats + [feat_iso]
                
                # 推理
                probs = clf.predict_proba([full_feats])[0]
                
                if job_idx not in image_scores:
                    image_scores[job_idx] = []
                image_scores[job_idx].append(probs)
            
            # --- Stage 5: 决策 (多鸟策略) & 执行操作 ---
            for job_idx, probs_list in image_scores.items():
                path = job_results[job_idx][0]
                filename = os.path.basename(path)
                
                # 取综合分最高的一只鸟代表整张图
                best_weighted_score = -1
                final_category = 0
                
                for probs in probs_list:
                    # 分数权重：Soft=1, Perfect=2
                    score = probs[1] * 1.0 + probs[2] * 2.0
                    if score > best_weighted_score:
                        best_weighted_score = score
                        final_category = np.argmax(probs)
                
                stats[final_category] += 1
                label_name = CLASS_MAP[final_category]
                
                # 模式 1: 写入 XMP
                if AUTO_WRITE_XMP:
                    XMPManager.write_tag(path, final_category)
                
                # 模式 2: 移动文件
                if AUTO_MOVE_FILES:
                    try:
                        dest = os.path.join(input_folder, label_name, filename)
                        shutil.move(path, dest)
                        # 伴随文件处理
                        xmp = os.path.splitext(path)[0] + ".xmp"
                        if os.path.exists(xmp):
                            shutil.move(xmp, os.path.join(input_folder, label_name, os.path.basename(xmp)))
                    except: pass
        
        pbar.update(len(batch_files))
        
    pbar.close()
    
    print("\n" + "="*60)
    print(f"🟣 Trash   (1星/紫色): {stats[0]}")
    print(f"🔴 Soft    (3星/红色): {stats[1]}")
    print(f"🟢 Perfect (5星/绿色): {stats[2]}")
    print("="*60)
    
    if AUTO_WRITE_XMP:
        print("✅ 标注完成！请在 Lightroom 中同步文件夹读取元数据。")
    if AUTO_MOVE_FILES:
        print("✅ 整理完成！文件已移动到对应子文件夹。")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()