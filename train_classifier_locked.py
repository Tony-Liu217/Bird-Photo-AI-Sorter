import os
import csv
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import linregress
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# 引入模块
from feature_extractor_dual import DualFeatureExtractor, ROI_STANDARD_SIZE
from detect_birds_multi import BirdDetector, load_best_available_image

try:
    from metadata_utils import get_iso_speed
except ImportError:
    def get_iso_speed(p): return 800

# ================= 配置 =================
BATCH_SIZE = 8
NUM_WORKERS = 8
CLASS_NAMES = ['Trash', 'Soft', 'Perfect']
MODEL_FILE = "best_bird_model_final.pkl"
ERROR_FILE = "error_analysis_final.csv"
# ========================================

def process_image_job(args):
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
        return (os.path.basename(path), iso, rois)
    except: return None

def main():
    print("=== AI 模型训练 (架构重构版) ===")
    
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="选择包含 labels_multi.csv 的文件夹")
    root.destroy()
    if not folder: return

    # 1. 读取 CSV
    csv_path = os.path.join(folder, "labels_multi.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(folder, "labels.csv")
        if not os.path.exists(csv_path): return

    print("解析标签...")
    labels_map = {}
    valid_files = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2: continue
            try:
                fname = row[0]
                if len(row) >= 3: b_idx, label = int(row[1]), int(row[2])
                else: b_idx, label = 0, int(row[1])
                
                # 映射: 1->0, 2->1, 3->2
                if label < 1 or label > 3: continue
                cls = label - 1
                
                if fname not in labels_map: labels_map[fname] = {}
                labels_map[fname][b_idx] = cls
                valid_files.add(fname)
            except: continue

    file_list = sorted(list(valid_files))
    print(f"有效图片: {len(file_list)}")

    detector = BirdDetector()
    extractor = DualFeatureExtractor() # 自动选择 GPU/CPU
    
    X, y, filenames = [], [], []
    
    # 2. 特征提取
    pbar = tqdm(total=len(file_list), desc="Feature Extraction")
    for i in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(folder, f) for f in batch_files]
        
        # IO 并行
        results_list = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_image_job, (p, detector)) for p in batch_paths]
            for f in futures:
                res = f.result()
                if res: results_list.append(res)
        
        # 准备数据
        batch_data = [] # [(roi, iso)]
        meta = [] 
        
        for fname, iso, rois in results_list:
            if fname not in labels_map: continue
            bird_dict = labels_map[fname]
            for idx, roi in enumerate(rois):
                if idx in bird_dict:
                    batch_data.append((roi, iso))
                    meta.append((bird_dict[idx], fname))
        
        # 批量计算
        if batch_data:
            feats_batch = extractor.extract_batch(batch_data)
            for k, f in enumerate(feats_batch):
                if f and any(f):
                    X.append(f)
                    y.append(meta[k][0])
                    filenames.append(meta[k][1])
        
        pbar.update(len(batch_files))
    pbar.close()
    
    if len(X) < 10: return

    # 3. 训练
    X_train, X_test, y_train, y_test, f_tr, f_te = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42
    )

    print("\n训练随机森林...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 4]}
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_probs = best_clf.predict_proba(X_test)
    
    print(f"最佳参数: {grid.best_params_}")
    print(f"🏆 测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 特征重要性
    names = ["Peak", "Mean", "Std", "Kurt", "Skew", "DoG", "SNR", "F_Mean", "F_Ratio", "Scale", "Color", "ISO"]
    imps = best_clf.feature_importances_
    print("\n📊 特征重要性:")
    for i in np.argsort(imps)[::-1]:
        if i < len(names): print(f"   {names[i]:<10}: {imps[i]:.4f}")

    joblib.dump(best_clf, MODEL_FILE)
    print(f"\n✅ 模型已保存: {MODEL_FILE}")

    # 4. 错误分析 & 回归
    errors = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            sev = "严重" if abs(y_test[i] - y_pred[i]) == 2 else "轻微"
            errors.append({
                'Filename': f_te[i], 'True': CLASS_NAMES[y_test[i]], 
                'Pred': CLASS_NAMES[y_pred[i]], 'Severity': sev, 'P_Soft': f"{y_probs[i][1]:.2f}"
            })
            
    if errors:
        with open(os.path.join(folder, ERROR_FILE), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Filename', 'True', 'Pred', 'Severity', 'P_Soft'])
            writer.writeheader()
            writer.writerows(errors)
        print(f"⚠️ {len(errors)} 个误判已导出")
        
    model_scores = np.sum(y_probs * np.array([0, 1, 2]), axis=1)
    slope, intercept, r_val, _, _ = linregress(y_test, model_scores)
    print(f"📈 线性回归 R²: {r_val**2:.4f}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()