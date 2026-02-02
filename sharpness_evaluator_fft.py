import cv2
import numpy as np

def calculate_sharpness_fft(image_roi, low_cut=0.05, high_cut=0.25):
    """
    基于 FFT (快速傅里叶变换) 的频域清晰度评分。
    包含"带通滤波"逻辑，专门对抗高感噪点。
    
    参数:
        image_roi: 输入图像 ROI (建议已标准化尺寸)
        low_cut:  低频截止比例 (0.0-1.0)。例如 0.05 表示屏蔽中心 5% 的低频区域(轮廓/光影)。
        high_cut: 高频截止比例 (0.0-1.0)。例如 0.25 表示只保留到 25% 的频率位置。
                  **关键点**: 设置得越低，越能过滤掉细碎的噪点；设置得越高，对极致细节越敏感。
                  针对鸟类摄影，0.2 - 0.4 是比较好的区间，能避开大部分 ISO 噪点。
    
    返回:
        score: 对数频谱能量的均值
    """
    if image_roi is None or image_roi.size == 0:
        return 0.0

    # 1. 转灰度
    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi

    # 2. FFT 变换
    rows, cols = gray.shape
    # 优化 FFT 计算速度 (填充到最优尺寸)
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = gray
    
    # 执行 2D FFT
    f = np.fft.fft2(nimg)
    # 将零频率分量移到频谱中心
    fshift = np.fft.fftshift(f)
    
    # 3. 计算频谱的幅度 (Magnitude)
    # 取对数是为了缩小动态范围，让人眼和算法更好处理
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 4. 构建带通滤波器 (Band-Pass Filter)
    # 我们需要一个环形的区域：切掉圆心(低频)，切掉最外圈(噪点)
    
    center_x, center_y = int(ncols/2), int(nrows/2)
    max_radius = min(center_x, center_y)
    
    # 转换坐标系为网格坐标
    y, x = np.ogrid[:nrows, :ncols]
    # 计算每个像素到中心的距离平方
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    
    # 定义半径阈值
    r_inner = int(max_radius * low_cut)  # 内圈半径
    r_outer = int(max_radius * high_cut) # 外圈半径
    
    # 创建环形掩膜 (Mask)
    # 只有在 r_inner 到 r_outer 之间的区域为 1，其余为 0
    mask = np.zeros((nrows, ncols), dtype=np.uint8)
    mask_area = (dist_sq >= r_inner**2) & (dist_sq <= r_outer**2)
    mask[mask_area] = 1
    
    # 5. 应用掩膜并计算平均能量
    # 只提取环形区域内的频谱能量
    band_pass_energy = magnitude_spectrum * mask
    
    # 计算非零区域的平均值
    # np.sum 计算总能量，np.count_nonzero 计算有效像素数
    total_energy = np.sum(band_pass_energy)
    pixel_count = np.count_nonzero(mask)
    
    if pixel_count == 0:
        return 0.0
        
    score = total_energy / pixel_count
    
    return score