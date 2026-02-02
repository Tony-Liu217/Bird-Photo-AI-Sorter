import os
import cv2
import csv
import tkinter as tk
from tkinter import filedialog
from detect_birds_multi import BirdDetector, load_best_available_image

# 必须与主程序保持一致
ROI_STANDARD_SIZE = 1600

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择【黄金校准集】文件夹")
    root.destroy()
    return path

def main():
    print("=== 多目标数据标记工具 (V3.1 含非鸟过滤) ===")
    print("操作指南:")
    print("  [0] 不是鸟 (Not Bird)  - 误识别物体，不参与训练")
    print("  [1] 完全废片 (Trash)")
    print("  [2] 部分失焦 (Soft)")
    print("  [3] 完美备选 (Perfect)")
    print("  [Space] 跳过当前鸟")
    print("  [ESC] 保存并退出")
    print("-" * 60)

    folder = select_folder()
    if not folder: return

    csv_file = os.path.join(folder, "labels_multi.csv")
    
    labeled_keys = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            for row in reader:
                if row and len(row) >= 3:
                    key = f"{row[0]}_{row[1]}" 
                    labeled_keys.add(key)
    
    print(f"已加载 {len(labeled_keys)} 条历史标记记录。")

    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
    file_list.sort()

    detector = BirdDetector()
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.path.getsize(csv_file) == 0:
            writer.writerow(["Filename", "Bird_Index", "Human_Label"])

        for filename in file_list:
            path = os.path.join(folder, filename)
            full_img = load_best_available_image(path)
            if full_img is None: continue

            rois = detector.detect_and_crop_all(full_img, standard_size=ROI_STANDARD_SIZE)
            
            if not rois:
                print(f"跳过 {filename}: 未检测到鸟类")
                continue

            for idx, roi in enumerate(rois):
                unique_key = f"{filename}_{idx}"
                if unique_key in labeled_keys:
                    continue

                h, w = roi.shape[:2]
                display_scale = 800 / max(h, w)
                display_img = cv2.resize(roi, (0, 0), fx=display_scale, fy=display_scale)
                
                info_text = f"{filename} | Bird {idx+1}/{len(rois)}"
                cv2.putText(display_img, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # 更新提示文字
                cv2.putText(display_img, "0:NotBird 1:Trash 2:Soft 3:Perfect", (10, h-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Multi-Labeler", display_img)
                
                valid_key = False
                label = None
                
                while not valid_key:
                    key = cv2.waitKey(0)
                    
                    if key == 27: # ESC
                        print("退出标记。")
                        return
                    elif key == 32: # Space
                        print(f"跳过当前目标。")
                        label = -1
                        valid_key = True
                    elif key == ord('0'): # [新增] 不是鸟
                        label = 0
                        valid_key = True
                    elif key == ord('1'):
                        label = 1
                        valid_key = True
                    elif key == ord('2'):
                        label = 2
                        valid_key = True
                    elif key == ord('3'):
                        label = 3
                        valid_key = True
                
                if label != -1:
                    status_str = "非鸟类" if label == 0 else str(label)
                    print(f"记录: {filename} [鸟{idx}] -> {status_str}")
                    writer.writerow([filename, idx, label])
                    f.flush()
                    labeled_keys.add(unique_key)

    cv2.destroyAllWindows()
    print(f"完成！数据已保存至 {csv_file}")

if __name__ == "__main__":
    main()