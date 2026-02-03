import numpy as np
import cv2
from scipy.stats import kurtosis, skew

# 尝试导入 PyTorch
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ================= 全局配置 =================
ROI_STANDARD_SIZE = 1600
BLOCK_SIZE = 16       
TOP_K_PERCENT = 0.05  
# ===========================================

class DualFeatureExtractor:
    """
    [核心组件] 双引擎特征提取器 (GPU + CPU Fallback)
    特征维度: 12维
    [Peak, Mean, Std, Kurt, Skew, DoG_Peak, SNR, FocusMean, FocusRatio, ScaleRatio, ColorVar, ISO]
    """
    def __init__(self):
        self.device = 'cpu'
        self.use_gpu = False
        
        if HAS_TORCH and torch.cuda.is_available():
            self.use_gpu = True
            self.device = 'cuda'
            print(f"⚡ 特征引擎初始化: GPU (CUDA Accelerated)")
            self._init_gpu_kernels()
        else:
            print(f"⚠️ 特征引擎初始化: CPU (OpenCV Mode)")

    def _init_gpu_kernels(self):
        """预加载 GPU 卷积核"""
        # Sobel
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        # Laplacian
        self.lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        # DoG
        k = torch.arange(5, dtype=torch.float32) - 2; k1 = torch.exp(-k**2 / 2); k1 /= k1.sum()
        self.gauss_k1 = (k1.view(1, 5) * k1.view(5, 1)).view(1, 1, 5, 5).to(self.device)
        k = torch.arange(9, dtype=torch.float32) - 4; k2 = torch.exp(-k**2 / 8); k2 /= k2.sum()
        self.gauss_k2 = (k2.view(1, 9) * k2.view(9, 1)).view(1, 1, 9, 9).to(self.device)

    def extract_batch(self, roi_list_with_iso):
        """
        入口函数
        输入: [(roi_numpy, iso_value), ...]
        输出: 特征矩阵 (List of lists)
        """
        if not roi_list_with_iso: return []
        
        if self.use_gpu:
            return self._extract_gpu(roi_list_with_iso)
        else:
            return self._extract_cpu(roi_list_with_iso)

    # ================= CPU 引擎 =================
    def _extract_cpu(self, data_list):
        features = []
        for roi, iso in data_list:
            if roi is None:
                features.append([0]*12); continue
            
            # 1. 预处理
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                img_rgb = roi.astype(np.float32) / 255.0
            else:
                gray = roi
                img_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            
            img_f = gray.astype(np.float32) / 255.0
            h, w = img_f.shape

            # 2. 掩膜
            _, mask_bin = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            erode_size = max(3, int(min(h, w) * 0.01))
            kernel = np.ones((erode_size, erode_size), np.uint8)
            inner_mask = cv2.erode(mask_bin, kernel, iterations=1)
            valid_mask = inner_mask > 0
            
            if np.sum(valid_mask) < 50:
                features.append([0]*12); continue

            # 3. 梯度
            gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            valid_grads = mag[valid_mask]

            # 4. DoG & Lap
            g1 = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.0)
            g2 = cv2.GaussianBlur(img_f, (0, 0), sigmaX=2.0)
            dog = np.abs(g1 - g2)
            lap = np.abs(cv2.Laplacian(img_f, cv2.CV_32F))
            
            # 5. 特征计算
            f_peak = np.percentile(valid_grads, 95) * 255
            f_mean = np.mean(valid_grads) * 255
            f_std = np.std(valid_grads) * 255
            f_kurt = kurtosis(valid_grads, fisher=True)
            f_skew = skew(valid_grads)
            
            valid_dog = dog[valid_mask]
            f_dog_p = np.percentile(valid_dog, 95) * 255
            f_snr = np.mean(valid_dog) / (np.mean(lap[valid_mask]) + 1e-9)
            
            # 6. 局部聚焦
            bs = BLOCK_SIZE
            sh, sw = h // bs, w // bs
            if sh > 0 and sw > 0:
                b_mag = cv2.resize(mag, (sw, sh), interpolation=cv2.INTER_AREA)
                b_msk = cv2.resize(inner_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
                v_blk = b_mag[b_msk > 0]
                if len(v_blk) > 0:
                    tk = max(1, int(len(v_blk) * TOP_K_PERCENT))
                    top_v = np.partition(v_blk, -tk)[-tk:]
                    f_fm = np.mean(top_v) * 255
                    f_fr = f_fm / (np.mean(v_blk) * 255 + 1e-6)
                else: f_fm, f_fr = 0, 0
            else: f_fm, f_fr = 0, 0
            
            # 7. 多尺度 & 色彩
            img_h = cv2.resize(img_f, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            gx_h = cv2.Sobel(img_h, cv2.CV_32F, 1, 0, ksize=3)
            gy_h = cv2.Sobel(img_h, cv2.CV_32F, 0, 1, ksize=3)
            mag_h = cv2.magnitude(gx_h, gy_h)
            mask_h = cv2.resize(inner_mask, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            valid_h = mag_h[mask_h > 0]
            f_sr = (np.mean(valid_h)*255)/(f_mean+1e-6) if len(valid_h)>0 else 0
            
            laps = [np.abs(cv2.Laplacian(img_rgb[:,:,c], cv2.CV_32F)) for c in range(3)]
            laps_std = np.std(np.stack(laps, axis=2), axis=2)
            valid_cv = laps_std[valid_mask]
            f_cv = np.mean(valid_cv) * 255 if len(valid_cv)>0 else 0
            
            f_iso = np.log10(max(iso, 50))
            
            features.append([f_peak, f_mean, f_std, f_kurt, f_skew, f_dog_p, f_snr, f_fm, f_fr, f_sr, f_cv, f_iso])
        return features

    # ================= GPU 引擎 =================
    def _extract_gpu(self, data_list):
        tensors = []
        isos = []
        for roi, iso in data_list:
            if roi is None:
                t = torch.zeros((3, ROI_STANDARD_SIZE, ROI_STANDARD_SIZE), dtype=torch.float32)
            else:
                t = torch.from_numpy(roi).float()
                if len(t.shape) == 3: t = t.permute(2, 0, 1)
                else: t = t.unsqueeze(0).repeat(3, 1, 1)
                h, w = t.shape[1], t.shape[2]
                pad_h, pad_w = ROI_STANDARD_SIZE - h, ROI_STANDARD_SIZE - w
                t = F.pad(t, (0, pad_w, 0, pad_h), "constant", 0)
            tensors.append(t)
            isos.append(iso)
            
        if not tensors: return []
        
        with torch.no_grad():
            input_rgb = torch.stack(tensors).to(self.device) / 255.0
            batch_size = input_rgb.shape[0]
            
            # 1. 预处理
            input_gray = 0.114 * input_rgb[:, 0:1] + 0.587 * input_rgb[:, 1:2] + 0.299 * input_rgb[:, 2:3]
            mask_bin = (input_gray > (10/255.0)).float()
            erode_k = int(ROI_STANDARD_SIZE * 0.01) | 1
            pad = erode_k // 2
            inner_mask = -F.max_pool2d(-mask_bin, kernel_size=erode_k, stride=1, padding=pad)
            
            # 2. 运算
            gx = F.conv2d(input_gray, self.sobel_x, padding=1)
            gy = F.conv2d(input_gray, self.sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2)
            
            g1 = F.conv2d(input_gray, self.gauss_k1, padding=2)
            g2 = F.conv2d(input_gray, self.gauss_k2, padding=4)
            dog = torch.abs(g1 - g2)
            lap = torch.abs(F.conv2d(input_gray, self.lap_kernel, padding=1))
            
            # 多尺度
            img_h = F.interpolate(input_gray, scale_factor=0.5, mode='bilinear')
            gx_h = F.conv2d(img_h, self.sobel_x, padding=1)
            gy_h = F.conv2d(img_h, self.sobel_y, padding=1)
            mag_h = torch.sqrt(gx_h**2 + gy_h**2)
            mask_h = F.interpolate(inner_mask, scale_factor=0.5, mode='nearest')
            
            # 色彩
            lap_k3 = self.lap_kernel.repeat(3, 1, 1, 1)
            lap_rgb = torch.abs(F.conv2d(input_rgb, lap_k3, padding=1, groups=3))
            lap_std = torch.std(lap_rgb, dim=1, keepdim=True)
            
            features = []
            for i in range(batch_size):
                curr_mask = inner_mask[i, 0]
                idx = curr_mask > 0.5
                if torch.sum(idx) < 50:
                    features.append([0]*12); continue
                
                v_mag = mag[i, 0][idx]
                v_dog = dog[i, 0][idx]
                v_lap = lap[i, 0][idx]
                
                # Stats
                f_peak = torch.quantile(v_mag, 0.95).item() * 255
                f_mean = torch.mean(v_mag).item() * 255
                f_std = torch.std(v_mag).item() * 255
                
                diff = v_mag - torch.mean(v_mag)
                std_v = torch.std(v_mag) + 1e-6
                f_skew = (torch.mean(diff**3)/std_v**3).item()
                f_kurt = (torch.mean(diff**4)/std_v**4).item()
                
                f_dog_p = torch.quantile(v_dog, 0.95).item() * 255
                f_snr = torch.mean(v_dog).item() / (torch.mean(v_lap).item() + 1e-9)
                
                # Focus
                bs = BLOCK_SIZE
                b_mag = F.avg_pool2d(mag[i], bs, stride=bs)
                b_msk = F.avg_pool2d(curr_mask.unsqueeze(0), bs, stride=bs)
                v_blk = b_mag[b_msk > 0.5]
                if v_blk.numel() > 0:
                    k = max(1, int(v_blk.numel() * TOP_K_PERCENT))
                    tk_v, _ = torch.topk(v_blk, k)
                    f_fm = torch.mean(tk_v).item() * 255
                    f_fr = f_fm / (torch.mean(v_blk).item() * 255 + 1e-6)
                else: f_fm, f_fr = 0, 0
                
                # Scale
                v_h = mag_h[i, 0][mask_h[i, 0] > 0.5]
                f_sr = (torch.mean(v_h).item()*255)/(f_mean+1e-6) if v_h.numel()>0 else 0
                
                # Color
                f_cv = torch.mean(lap_std[i, 0][idx]).item() * 255
                
                f_iso = np.log10(max(isos[i], 50))
                
                features.append([f_peak, f_mean, f_std, f_kurt, f_skew, f_dog_p, f_snr, f_fm, f_fr, f_sr, f_cv, f_iso])
            
            return features