import torch
import torch.nn.functional as F
import numpy as np

class GPUFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        # 预先定义 Sobel 核，避免每次重复创建
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # 预先定义腐蚀核 (3x3 全1)
        self.erode_kernel = torch.ones((1, 1, 3, 3), device=device)

    def to_tensor(self, img_np):
        """将 OpenCV (H, W, C) 转为 PyTorch (1, 1, H, W) 灰度张量"""
        if img_np is None: return None
        # 归一化到 0-1 并转为 float32
        tensor = torch.from_numpy(img_np).float().to(self.device) / 255.0
        
        if len(tensor.shape) == 3: # H,W,C -> 1,H,W
            # 简单的 RGB 转灰度 (0.299R + 0.587G + 0.114B)
            # 这里简化处理，直接取平均或使用加权
            gray = 0.299 * tensor[:,:,2] + 0.587 * tensor[:,:,1] + 0.114 * tensor[:,:,0]
        else:
            gray = tensor
            
        return gray.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    def extract(self, roi_np):
        if roi_np is None: return None
        
        # 1. 数据上传 GPU
        img_t = self.to_tensor(roi_np)
        B, C, H, W = img_t.shape

        # 2. 生成内部掩膜 (GPU 版腐蚀)
        # 二值化
        mask = (img_t > 0.004).float() # 1/255 约为 0.0039
        
        # 动态计算腐蚀次数
        erode_steps = max(1, int(min(H, W) * 0.01))
        # 使用负池化或卷积模拟腐蚀，这里简化用卷积
        # 或者直接复用 Mask（因为 YOLO 分割已经很准了，这里可以适当简化）
        # 为了速度，我们这里简单地向内收缩：
        # 使用 max_pool2d 的逆操作 (min_pool) 来实现腐蚀
        inner_mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        # 如果需要更强腐蚀，多做几次或增大 kernel
        
        # 3. 计算梯度 (Sobel on GPU)
        gx = F.conv2d(img_t, self.sobel_x, padding=1)
        gy = F.conv2d(img_t, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # 只取掩膜内的梯度
        valid_grads = magnitude[inner_mask > 0.5]
        
        if valid_grads.numel() == 0: return [0]*6

        # 特征 A: 峰值梯度 (95%)
        # torch.quantile 相当于 numpy.percentile
        feat_peak = torch.quantile(valid_grads, 0.95).item() * 255
        
        # 特征 B: 平均梯度
        feat_mean = torch.mean(valid_grads).item() * 255
        
        # 特征 C: 标准差
        feat_std = torch.std(valid_grads).item() * 255

        # 4. FFT 频域分析 (GPU FFT)
        # PyTorch 的 FFT 极快
        fft_res = torch.fft.fft2(img_t)
        fft_shift = torch.fft.fftshift(fft_res)
        mag_spec = 20 * torch.log(torch.abs(fft_shift) + 1)
        
        # 构建环形 Mask (向量化操作)
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=self.device).view(-1, 1) - cy
        x = torch.arange(W, device=self.device).view(1, -1) - cx
        dist_sq = x**2 + y**2
        max_r = min(cy, cx)
        
        # Low Cut / Mid / High 定义
        r2_low = (max_r * 0.10)**2
        r2_mid = (max_r * 0.30)**2
        r2_high = (max_r * 0.70)**2
        
        # 提取频段能量
        def calc_band_energy(r_inner_sq, r_outer_sq):
            mask_band = (dist_sq >= r_inner_sq) & (dist_sq <= r_outer_sq)
            vals = mag_spec.squeeze()[mask_band]
            if vals.numel() == 0: return 0.0
            return torch.mean(vals).item()

        feat_fft_mid = calc_band_energy(r2_low, r2_mid)
        feat_fft_high = calc_band_energy(r2_mid, r2_high)

        # 5. 亮度
        valid_pixels = img_t[inner_mask > 0.5]
        feat_bright = torch.mean(valid_pixels).item() * 255 if valid_pixels.numel() > 0 else 0

        # 清理显存 (可选，但在循环中建议)
        # torch.cuda.empty_cache() 
        
        return [feat_peak, feat_mean, feat_std, feat_fft_mid, feat_fft_high, feat_bright]

# 使用示例 (在主程序中)：
# extractor = GPUFeatureExtractor()
# feats = extractor.extract(roi)