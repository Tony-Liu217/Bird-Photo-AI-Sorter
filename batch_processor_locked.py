import os
import shutil
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 引入模块
from detect_birds_multi import BirdDetector, load_best_available_image
from feature_extractor_dual import DualFeatureExtractor, ROI_STANDARD_SIZE

try:
    from metadata_utils import get_iso_speed
except ImportError:
    def get_iso_speed(p): return 800

# ================= 配置 =================
BATCH_SIZE = 4
NUM_WORKERS = 8
MODEL_FILE = "best_bird_model_final.pkl" # 对应 train 生成的文件
CLASS_MAP = {0: 'Trash', 1: 'Soft', 2: 'Perfect'}
AUTO_WRITE_XMP = False
AUTO_MOVE_FILES = False
# ========================================

class XMPManager:
    @staticmethod
    def write_tag(image_path, category_idx):
        xmp_path = os.path.splitext(image_path)[0] + ".xmp"
        if category_idx == 0: label, rating = "紫色", 1
        elif category_idx == 1: label, rating = "红色", 3
        elif category_idx == 2: label, rating = "绿色", 5
        else: label, rating = "黄色", 0
        
        xml = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns/"><rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}" xmp:Label="{label}"/></rdf:RDF></x:xmpmeta>"""
        try:
            with open(xmp_path, 'w', encoding='utf-8') as f: f.write(xml)
        except: pass

def process_io_job(args):
    path, detector = args
    try:
        full_img = load_best_available_image(path)
        if full_img is None: return None
        iso = get_iso_speed(path)
        if hasattr(detector, 'detect_and_crop_all'):
            rois = detector.detect_and_crop_all(full_img, standard_size=ROI_STANDARD_SIZE)
        else:
            roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
            rois = [roi] if roi else []
        return (path, iso, rois)
    except: return None

def select_folder():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askdirectory(title="选择照片文件夹")
    root.destroy()
    return path

def main():
    print("=== 🦅 Bird AI Sorter (Universal Edition) ===")
    
    global AUTO_MOVE_FILES, AUTO_WRITE_XMP
    print("\n请选择模式:")
    print("  [1] 整理模式 (移动文件)")
    print("  [2] 标注模式 (生成 XMP)")
    while True:
        mode = input("输入 1 或 2: ").strip()
        if mode == '1': AUTO_MOVE_FILES = True; AUTO_WRITE_XMP = False; break
        elif mode == '2': AUTO_MOVE_FILES = False; AUTO_WRITE_XMP = True; break

    if not os.path.exists(MODEL_FILE):
        print(f"❌ 错误: 未找到模型 {MODEL_FILE}")
        return
    clf = joblib.load(MODEL_FILE)
    print("✅ 模型加载完成")

    folder = select_folder()
    if not folder: return

    if AUTO_MOVE_FILES:
        for n in CLASS_MAP.values(): os.makedirs(os.path.join(folder, n), exist_ok=True)
        os.makedirs(os.path.join(folder, "Unidentified"), exist_ok=True)

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.dng', '.orf', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
    file_list.sort()
    
    print(f"处理 {len(file_list)} 张照片...")
    
    detector = BirdDetector()
    extractor = DualFeatureExtractor()
    stats = {0: 0, 1: 0, 2: 0, 'Unidentified': 0}

    for i in tqdm(range(0, len(file_list), BATCH_SIZE), desc="Processing"):
        batch_files = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(folder, f) for f in batch_files]
        
        # IO
        results = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_io_job, (p, detector)) for p in batch_paths]
            for f in futures:
                res = f.result()
                if res: results.append(res)
        
        # Batch Prepare
        batch_data = []
        meta = [] 
        for path, iso, rois in results:
            job_idx = batch_paths.index(path)
            for roi in rois:
                batch_data.append((roi, iso))
                meta.append(job_idx)
        
        # Feature & Predict
        if batch_data:
            feats_batch = extractor.extract_batch(batch_data)
            
            img_scores = {}
            for k, f in enumerate(feats_batch):
                if f and any(f):
                    probs = clf.predict_proba([f])[0]
                    j_idx = meta[k]
                    if j_idx not in img_scores: img_scores[j_idx] = []
                    img_scores[j_idx].append(probs)
            
            for j_idx, probs_list in img_scores.items():
                path = batch_paths[j_idx]
                fname = os.path.basename(path)
                
                best_score = -1
                final_cat = 0
                for p in probs_list:
                    s = p[1]*1.0 + p[2]*2.0 
                    if s > best_score:
                        best_score = s
                        final_cat = np.argmax(p)
                
                label_name = CLASS_MAP[final_cat]
                stats[final_cat] += 1
                
                if AUTO_WRITE_XMP: XMPManager.write_tag(path, final_cat)
                if AUTO_MOVE_FILES:
                    try:
                        shutil.move(path, os.path.join(folder, label_name, fname))
                        xmp = os.path.splitext(path)[0] + ".xmp"
                        if os.path.exists(xmp):
                            shutil.move(xmp, os.path.join(folder, label_name, os.path.basename(xmp)))
                    except: pass
        
    print("\n" + "="*60)
    print(f"🟣 Trash: {stats[0]} | 🔴 Soft: {stats[1]} | 🟢 Perfect: {stats[2]}")
    print("="*60)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()