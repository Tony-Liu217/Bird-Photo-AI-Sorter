from ultralytics import YOLO
import cv2
import numpy as np
import os
import rawpy
import torch # 需要导入 torch 来检测显卡

# ================= 配置区域 =================
# 使用实例分割模型 (Segmentation)
# 相比普通检测模型，它能精准勾勒鸟的轮廓，从而切出更干净的 ROI
MODEL_PATH = 'yolov8x-seg.pt' 
CONFIDENCE_THRESHOLD = 0.15 
INFERENCE_SIZE = 1280 
# ===========================================

def load_best_available_image(path):
    """
    [通用工具] 获取文件能提供的最高画质图像数据。
    支持 RAW (内嵌JPG) 和普通图片。
    """
    if not os.path.exists(path): return None
    ext = os.path.splitext(path)[1].lower()
    raw_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.rw2'}
    
    try:
        if ext in raw_exts:
            with rawpy.imread(path) as raw:
                try:
                    # 尝试提取内嵌预览图 (针对 Z8/Z9 等新机型)
                    thumb = raw.extract_thumb()
                except rawpy.LibRawError:
                    return None
                
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return None
        else:
            # 普通图片读取 (支持中文路径)
            img_array = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"加载异常 {os.path.basename(path)}: {e}")
        return None

class BirdDetector:
    def __init__(self):
        # 1. 自动检测运行设备 (GPU 优先)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            device_name = torch.cuda.get_device_name(0)
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"🚀 运行模式: GPU ({device_name})")
        else:
            self.device = 'cpu'
            print(f"正在加载分割 AI 模型: {MODEL_PATH}")
            print(f"⚠️ 运行模式: CPU (未检测到可用显卡)")
        
        # 加载 YOLO 模型
        self.model = YOLO(MODEL_PATH)
        self.target_class_id = 14 # 鸟类类别ID

    def detect_and_crop(self, high_res_img, output_scale=1.0, standard_size=None, shrink_ratio=0.0):
        """
        核心检测与裁切函数
        
        参数:
            high_res_img: 全分辨率原图
            output_scale: 相对缩放比例 (旧参数)
            standard_size: 统一长边像素值 (新参数，推荐 1600)
            shrink_ratio: 向内收缩比例 (使用分割模型时，此参数通常设为0，因为mask已经很紧致了)
        """
        if high_res_img is None: return None, None

        # 1. 降采样推理 (为了速度，将大图缩小喂给 AI)
        h, w = high_res_img.shape[:2]
        scale_factor = INFERENCE_SIZE / max(h, w)
        
        if scale_factor < 1:
            inference_img = cv2.resize(high_res_img, (int(w * scale_factor), int(h * scale_factor)))
        else:
            inference_img = high_res_img
            scale_factor = 1.0

        # 2. AI 识别
        # 自动判断半精度: GPU开启(True)提速, CPU关闭(False)防报错
        use_half = (self.device != 'cpu')

        results = self.model(
            inference_img, 
            verbose=False, 
            agnostic_nms=True,
            device=self.device, 
            half=use_half,       
            retina_masks=True    # 开启高精度 Mask
        )
        
        # 获取最佳结果 (包含 Box 和 Mask)
        box_small, mask_segments = self._get_best_bird_data(results)
        
        if box_small is None: return None, None

        # 3. 智能坐标优化 (Segmentation Logic)
        # 如果存在 Mask 轮廓，使用 Mask 的极限边界来替代预测框
        # 这样能切掉 YOLO 预测框中多余的背景空隙
        if mask_segments is not None and len(mask_segments) > 0:
            segments = mask_segments
            # segments 是 float 类型的像素坐标
            min_x = np.min(segments[:, 0])
            min_y = np.min(segments[:, 1])
            max_x = np.max(segments[:, 0])
            max_y = np.max(segments[:, 1])
            
            # 使用 Mask 计算出的紧致框
            box_small = [min_x, min_y, max_x, max_y]

        # 4. 映射回原图坐标 (Upscaling)
        x1, y1, x2, y2 = box_small
        
        real_x1 = max(0, int(x1 / scale_factor))
        real_y1 = max(0, int(y1 / scale_factor))
        real_x2 = min(w, int(x2 / scale_factor))
        real_y2 = min(h, int(y2 / scale_factor))
        
        box_in_original = [real_x1, real_y1, real_x2, real_y2]

        # 5. 裁切 ROI
        roi = high_res_img[real_y1:real_y2, real_x1:real_x2]
        if roi.size == 0: return None, None

        # 6. 尺寸标准化 (标准化分辨率以保证评分公平)
        if standard_size is not None and standard_size > 0:
            rh, rw = roi.shape[:2]
            max_dim = max(rh, rw)
            
            # 缩放到标准尺寸 (例如长边 1600)
            resize_scale = standard_size / max_dim
            new_w = int(rw * resize_scale)
            new_h = int(rh * resize_scale)
            
            # 智能插值：缩小用 AREA，放大用 CUBIC
            interp = cv2.INTER_AREA if resize_scale < 1 else cv2.INTER_CUBIC
            roi = cv2.resize(roi, (new_w, new_h), interpolation=interp)
            
        elif output_scale != 1.0:
            # 旧逻辑兼容
            nw = int(roi.shape[1] * output_scale)
            nh = int(roi.shape[0] * output_scale)
            roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

        return roi, box_in_original

    def _get_best_bird_data(self, results):
        """同时获取最佳的 Box 和对应的 Mask"""
        best_box = None
        best_segments = None
        max_conf = 0
        
        for result in results:
            if result.boxes is None: continue
            
            boxes = result.boxes
            masks = result.masks # 分割模型独有
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 筛选置信度
                if cls_id == self.target_class_id and conf > CONFIDENCE_THRESHOLD:
                    if conf > max_conf:
                        max_conf = conf
                        # 获取预测框 (注意：GPU模式下需转回CPU)
                        best_box = box.xyxy[0].cpu().numpy().astype(float)
                        
                        # 获取 Mask 轮廓点 (如果有)
                        if masks is not None:
                            try:
                                # masks.xy 返回的是像素坐标列表
                                best_segments = masks.xy[i]
                            except:
                                best_segments = None
                                
        return best_box, best_segments

if __name__ == "__main__":
    pass