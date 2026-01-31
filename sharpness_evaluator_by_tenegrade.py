import cv2
import numpy as np

def calculate_sharpness_tenengrad(image_roi, sensitivity=3, top_percentile=0.4):
    """
    Tenengrad 评分算法 (V2.0 优化版)
    
    改进点：
    不再计算全图平均分，而是计算"最锐利的前 N% 像素"的平均分。
    这能有效解决"鸟太小导致背景拉低总分"的问题。
    
    参数:
        image_roi: 鸟类区域图像
        sensitivity: 幂律变换指数 (建议改回 2，因为 Percentile 逻辑已经足够过滤背景)
        top_percentile: 取前百分之多少的像素 (0.1 表示前 10%)
    """
    if image_roi is None or image_roi.size == 0:
        return 0.0

    # 1. 转灰度
    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi

    # 2. 计算梯度 (使用 64位浮点数防止溢出)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 3. 计算梯度幅值
    magnitude = np.sqrt(gx**2 + gy**2)

    # 4. 归一化 (防溢出)
    max_mag = np.max(magnitude)
    if max_mag == 0: return 0.0
    magnitude_norm = magnitude / max_mag

    # 5. 非线性拉伸 (Sensitivity = 2 也就是平方，比较温和)
    magnitude_powered = magnitude_norm ** sensitivity

    # ================= 核心改进 =================
    # 6. Top Percentile 逻辑
    # 我们不关心那些平滑的羽毛或虚化的背景，只关心最锐利的线条
    
    # 展平数组并排序
    sorted_pixels = np.sort(magnitude_powered.flatten())
    
    # 计算我们要取多少个像素 (比如总共100个点，取前10个)
    num_pixels = int(len(sorted_pixels) * top_percentile)
    
    # 避免像素太少导致报错
    if num_pixels < 1:
        num_pixels = 1
        
    # 取最后 N 个像素 (因为是升序排列，最大的在最后)
    top_pixels = sorted_pixels[-num_pixels:]
    
    # 计算这些精英像素的平均分
    # 乘 1000 依然是为了让分数好看
    score = np.mean(top_pixels) * 1000

    return score