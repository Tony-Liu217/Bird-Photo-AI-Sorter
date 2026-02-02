import os
import cv2
import csv
import tkinter as tk
from tkinter import filedialog
from detect_birds_single_maskenabled import BirdDetector, load_best_available_image

# 必须与主程序保持一致
ROI_STANDARD_SIZE = 1600

def select_folder():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="选择【黄金校准集】文件夹")
    root.destroy()
    return path

def main():
    print("=== 快速数据标记工具 (三级梯度版) ===")
    print("操作指南:")
    print("  [1] 完全废片 (Trash)   - 彻底糊掉/严重跑焦")
    print("  [2] 部分失焦 (Soft)    - 缩图可用/轻微肉/噪点重")
    print("  [3] 完美备选 (Perfect) - 刀锐奶化/数毛级")
    print("  [ESC] 保存并退出")
    print("-" * 60)

    folder = select_folder()
    if not folder: return

    # 准备 CSV 文件记录结果
    csv_file = os.path.join(folder, "labels.csv")
    
    # 读取已有的标记（支持断点续传）
    labeled_files = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            for row in reader:
                if row: labeled_files.add(row[0])
    
    print(f"已标记: {len(labeled_files)} 张，继续处理剩余图片...")

    # 扫描文件
    valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.jpg', '.jpeg', '.png'}
    file_list = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
    file_list.sort()

    detector = BirdDetector()
    
    # 打开 CSV 准备追加写入
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 如果是新文件，写入表头
        if len(labeled_files) == 0:
            writer.writerow(["Filename", "Human_Label"])

        for filename in file_list:
            if filename in labeled_files:
                continue

            path = os.path.join(folder, filename)
            
            # 1. 加载图片
            full_img = load_best_available_image(path)
            if full_img is None: continue

            # 2. 获取 ROI (这一点至关重要，必须看算法看到的东西)
            roi, _ = detector.detect_and_crop(full_img, standard_size=ROI_STANDARD_SIZE)
            
            if roi is None:
                print(f"跳过 {filename}: 未检测到鸟类")
                continue

            # 3. 显示图片并等待按键
            # 缩小一点以便显示
            h, w = roi.shape[:2]
            display_scale = 800 / max(h, w)
            display_img = cv2.resize(roi, (0, 0), fx=display_scale, fy=display_scale)
            
            # 在图片上显示文件名
            cv2.putText(display_img, f"{filename}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "1:Trash  2:Soft  3:Perfect", (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Labeler (Press 1/2/3)", display_img)
            
            valid_key = False
            while not valid_key:
                key = cv2.waitKey(0)
                
                if key == 27: # ESC
                    print("退出标记。")
                    return
                elif key == ord('1'):
                    label = 1
                    valid_key = True
                elif key == ord('2'):
                    label = 2
                    valid_key = True
                elif key == ord('3'):
                    label = 3
                    valid_key = True
            
            # 4. 写入结果
            print(f"标记 {filename}: {label}")
            writer.writerow([filename, label])
            f.flush() # 立即保存防止崩溃丢失

    cv2.destroyAllWindows()
    print("所有图片处理完成！结果已保存在 labels.csv")

if __name__ == "__main__":
    main()