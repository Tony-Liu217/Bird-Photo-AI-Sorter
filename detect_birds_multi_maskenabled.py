from ultralytics import YOLO
import cv2
import numpy as np
import os
import rawpy
import torch  # 必须导入 torch 用于检测显卡

# ================= 配置区域 =================
MODEL_PATH = 'yolov8x-seg.pt' 
CONFIDENCE_THRESHOLD = 0.15 
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
    except Exception as e:
        print(f"加载异常 {os.path.basename(path)}: {e}")
        return None

class BirdDetector:
    def __init__(self):
        # 1. 恢复自动检测 (不再强制 CPU)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"🚀 运行模式: GPU ({gpu_name}) | FP16半精度: 开启")
        else:
            self.device = 'cpu'
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"⚠️ 运行模式: CPU (未检测到 PyTorch 兼容显卡)")
        
        self.model = YOLO(MODEL_PATH)
        self.target_class_id = 14 

    def detect_and_crop(self, high_res_img, output_scale=1.0, standard_size=None, shrink_ratio=0.0, mask_background=True):
        if high_res_img is None: return None, None

        # 1. 降采样推理
        h, w = high_res_img.shape[:2]
        scale_factor = INFERENCE_SIZE / max(h, w)
        
        if scale_factor < 1:
            inference_img = cv2.resize(high_res_img, (int(w * scale_factor), int(h * scale_factor)))
        else:
            inference_img = high_res_img
            scale_factor = 1.0

        # 2. AI 识别
        # 动态决定是否开启半精度 (GPU开，CPU关)
        use_half = (self.device != 'cpu')

        results = self.model(
            inference_img, 
            verbose=False, 
            agnostic_nms=True,
            device=self.device, 
            half=use_half,       # GPU 下开启 FP16 提速
            retina_masks=True
        )
        
        # 获取最佳结果
        box_small, mask_segments = self._get_best_bird_data(results)
        
        if box_small is None: return None, None

        # 3. 智能坐标优化 & 蒙版生成
        if mask_segments is not None and len(mask_segments) > 0:
            segments = mask_segments
            
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
                
                # 腐蚀遮罩
                erosion_size = max(3, int(min(h, w) * 0.002)) 
                kernel = np.ones((erosion_size, erosion_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                
                high_res_img = cv2.bitwise_and(high_res_img, high_res_img, mask=mask)

        # 4. 映射回原图坐标
        x1, y1, x2, y2 = box_small
        
        real_x1 = max(0, int(x1 / scale_factor))
        real_y1 = max(0, int(y1 / scale_factor))
        real_x2 = min(w, int(x2 / scale_factor))
        real_y2 = min(h, int(y2 / scale_factor))
        
        box_in_original = [real_x1, real_y1, real_x2, real_y2]

        # 5. 裁切 ROI
        roi = high_res_img[real_y1:real_y2, real_x1:real_x2]
        
        if roi.size == 0: return None, None

        # 6. 尺寸标准化
        if standard_size is not None and standard_size > 0:
            rh, rw = roi.shape[:2]
            max_dim = max(rh, rw)
            resize_scale = standard_size / max_dim
            new_w = int(rw * resize_scale)
            new_h = int(rh * resize_scale)
            interp = cv2.INTER_AREA if resize_scale < 1 else cv2.INTER_CUBIC
            roi = cv2.resize(roi, (new_w, new_h), interpolation=interp)
        elif output_scale != 1.0:
            nw = int(roi.shape[1] * output_scale)
            nh = int(roi.shape[0] * output_scale)
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

        return roi, box_in_original

    def _get_best_bird_data(self, results):
        best_box = None
        best_segments = None
        max_conf = 0
        
        for result in results:
            if result.boxes is None: continue
            boxes = result.boxes
            masks = result.masks
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == self.target_class_id and conf > CONFIDENCE_THRESHOLD:
                    if conf > max_conf:
                        max_conf = conf
                        # 关键：GPU 模式下 tensor 在显存里，必须 .cpu() 搬回内存才能转 numpy
                        best_box = box.xyxy[0].cpu().numpy().astype(float)
                        if masks is not None:
                            try:
                                best_segments = masks.xy[i]
                            except:
                                best_segments = None
        return best_box, best_segments

if __name__ == "__main__":
    pass