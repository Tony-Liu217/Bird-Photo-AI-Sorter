from ultralytics import YOLO
import cv2
import numpy as np
import os
import rawpy
import torch

# ================= 配置区域 =================
MODEL_PATH = 'yolov8x-seg.pt' 
CONFIDENCE_THRESHOLD = 0.30 # 提高阈值，过滤非鸟类误判 (原 0.15)
INFERENCE_SIZE = 1280 
# ===========================================

def load_best_available_image(path):
    """[通用工具] 获取文件能提供的最高画质图像数据"""
    if not os.path.exists(path): return None
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2'}
    
    try:
        if ext in raw_exts:
            with rawpy.imread(path) as raw:
                try:
                    thumb = raw.extract_thumb()
                except rawpy.LibRawError:
                    return None
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return None
        else:
            img_array = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

class BirdDetector:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"🚀 运行模式: GPU ({gpu_name}) | FP16半精度: 开启")
        else:
            self.device = 'cpu'
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"⚠️ 运行模式: CPU")
        
        self.model = YOLO(MODEL_PATH)
        self.target_class_id = 14 

    def detect_and_crop_all(self, high_res_img, standard_size=None, mask_background=True, mask_shrink_ratio=0.005):
        """
        [多目标版] 检测并返回所有鸟类的 ROI
        
        参数:
            mask_shrink_ratio (float): 遮罩向内收缩的比率 (相对于图片短边)。
                                       0.005 (0.5%) 是默认值，能去除细微边缘。
                                       如果背景残留严重，尝试提高到 0.01 或 0.02。
        返回: list of ROI images
        """
        if high_res_img is None: return []

        h, w = high_res_img.shape[:2]
        scale_factor = INFERENCE_SIZE / max(h, w)
        
        if scale_factor < 1:
            inference_img = cv2.resize(high_res_img, (int(w * scale_factor), int(h * scale_factor)))
        else:
            inference_img = high_res_img
            scale_factor = 1.0

        use_half = (self.device != 'cpu')
        results = self.model(
            inference_img, 
            verbose=False, 
            agnostic_nms=True,
            device=self.device, 
            half=use_half,       
            retina_masks=True
        )
        
        # 获取所有鸟类数据
        all_detections = self._get_all_bird_data(results)
        
        if not all_detections: return []

        final_rois = []

        # 遍历每一只识别到的鸟
        for (box_small, segments) in all_detections:
            
            # 1. 智能坐标优化 & 蒙版生成
            current_high_res = high_res_img.copy() # 必须copy，否则多只鸟会互相涂黑
            
            if segments is not None and len(segments) > 0:
                # (A) 计算紧致框
                min_x = np.min(segments[:, 0])
                min_y = np.min(segments[:, 1])
                max_x = np.max(segments[:, 0])
                max_y = np.max(segments[:, 1])
                box_small = [min_x, min_y, max_x, max_y]
                
                # (B) 像素级抠图
                if mask_background:
                    segments_high_res = segments / scale_factor
                    segments_high_res = segments_high_res.astype(np.int32)
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [segments_high_res], 255)
                    
                    # === 核心修改：动态强力腐蚀 ===
                    if mask_shrink_ratio > 0:
                        # 动态计算核大小 (基于图像短边)
                        # 例如 4000px 的图，ratio 0.005 -> 收缩 20px
                        erosion_size = int(min(h, w) * mask_shrink_ratio)
                        # 保证最小为3，且为奇数
                        erosion_size = max(3, erosion_size)
                        if erosion_size % 2 == 0: erosion_size += 1
                        
                        kernel = np.ones((erosion_size, erosion_size), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                    
                    current_high_res = cv2.bitwise_and(current_high_res, current_high_res, mask=mask)

            # 2. 映射回原图坐标
            x1, y1, x2, y2 = box_small
            real_x1 = max(0, int(x1 / scale_factor))
            real_y1 = max(0, int(y1 / scale_factor))
            real_x2 = min(w, int(x2 / scale_factor))
            real_y2 = min(h, int(y2 / scale_factor))
            
            # 3. 裁切 ROI
            roi = current_high_res[real_y1:real_y2, real_x1:real_x2]
            
            if roi.size == 0: continue

            # 4. 尺寸标准化
            if standard_size is not None and standard_size > 0:
                rh, rw = roi.shape[:2]
                max_dim = max(rh, rw)
                resize_scale = standard_size / max_dim
                new_w = int(rw * resize_scale)
                new_h = int(rh * resize_scale)
                interp = cv2.INTER_AREA if resize_scale < 1 else cv2.INTER_CUBIC
                roi = cv2.resize(roi, (new_w, new_h), interpolation=interp)

            final_rois.append(roi)

        return final_rois

    def _get_all_bird_data(self, results):
        """返回列表中所有的鸟类 (box, mask) 元组"""
        detections = []
        
        for result in results:
            if result.boxes is None: continue
            boxes = result.boxes
            masks = result.masks
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == self.target_class_id and conf > CONFIDENCE_THRESHOLD:
                    # GPU Tensor -> CPU Numpy
                    best_box = box.xyxy[0].cpu().numpy().astype(float)
                    best_segments = None
                    if masks is not None:
                        try:
                            # 同样确保转回 CPU
                            best_segments = masks.xy[i]
                        except: pass
                    
                    detections.append((best_box, best_segments))
                    
        return detections

if __name__ == "__main__":
    pass