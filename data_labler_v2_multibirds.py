import os
import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 引入核心模块
try:
    from detect_birds_multi import BirdDetector, load_best_available_image
except ImportError:
    from detect_birds_multi import BirdDetector, load_best_available_image

# ================= 配置 =================
ROI_STANDARD_SIZE = 1600
LABEL_FILE = "labels_multi.csv"
ERROR_FILES = ["error_analysis_v9.csv", "error_analysis.csv"]
# =======================================

class Visualizer:
    """UI 视觉增强器"""
    
    COLORS = {
        'Trash': (128, 128, 255),   # 红/紫
        'Soft': (0, 255, 255),      # 黄
        'Perfect': (0, 255, 0),     # 绿
        'Unknown': (200, 200, 200), # 灰
        'Text': (255, 255, 255),    # 白
        'Bg': (30, 30, 30)          # 深灰背景
    }

    @staticmethod
    def add_hud(image, filename, bird_idx, total_birds, human_label=None, ai_pred=None, ai_conf=None, mode="New"):
        h, w = image.shape[:2]
        
        # 1. 缩放图片适应屏幕 (限制最大边长 800px)
        scale = 1.0
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        else:
            new_w, new_h = w, h
            
        # 2. 创建画布
        top_border = 40
        bottom_border = 60
        canvas_h = new_h + top_border + bottom_border
        canvas = np.zeros((canvas_h, new_w, 3), dtype=np.uint8)
        canvas[:] = Visualizer.COLORS['Bg'] 
        
        canvas[top_border:top_border+new_h, 0:new_w] = image
        
        # 3. 顶部信息
        title = f"FILE: {filename} | Bird {bird_idx+1}/{total_birds}"
        cv2.putText(canvas, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Visualizer.COLORS['Text'], 1)
        
        mode_color = (0, 165, 255) if mode == "Correction" else (255, 200, 0)
        cv2.putText(canvas, f"MODE: {mode}", (new_w - 180, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # 4. 底部对比信息
        y_text = canvas_h - 20
        
        # 左侧：人工标记
        human_str = str(human_label) if human_label is not None else "None"
        h_color = Visualizer.COLORS.get(human_str, Visualizer.COLORS['Unknown'])
        cv2.putText(canvas, "HUMAN:", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Visualizer.COLORS['Text'], 1)
        cv2.putText(canvas, human_str, (90, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, h_color, 2)

        # 右侧：AI 预测 (仅在有数据时显示)
        if ai_pred and str(ai_pred) != "nan":
            ai_str = str(ai_pred)
            a_color = Visualizer.COLORS.get(ai_str, Visualizer.COLORS['Unknown'])
            
            conf_str = f"{ai_conf:.1%}" if ai_conf is not None else ""
            text = f"AI: {ai_str} {conf_str}"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(canvas, text, (new_w - tw - 20, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, a_color, 1)
            
            # 不一致警告
            if human_label and str(human_label) != "None" and str(human_label) != str(ai_pred):
                cv2.putText(canvas, "[DIFF]", (new_w//2 - 30, y_text), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return canvas

def select_folder():
    root = tk.Tk(); root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title="选择工作目录")
    root.destroy()
    return path

def main():
    print("=== 🦅 鸟类摄影数据闭环工具 (V5.0 多目标整合版) ===")
    
    folder = select_folder()
    if not folder: return

    label_csv_path = os.path.join(folder, LABEL_FILE)
    
    # 自动查找存在的误差文件
    error_csv_path = None
    for ef in ERROR_FILES:
        p = os.path.join(folder, ef)
        if os.path.exists(p):
            error_csv_path = p
            break

    print("\n请选择模式:")
    print("  [1] 新图标记 (Label New): 遍历文件夹中未标记的图片")
    print("  [2] 纠错模式 (Correct): 读取 error_analysis.csv，复核误判")
    mode_input = input("输入 1 或 2: ").strip()
    
    work_items = [] # list of dict: {'Filename', 'Bird_Index', ...}
    current_mode = "New"

    # ================= 模式 2: 纠错模式 =================
    if mode_input == '2':
        if not error_csv_path:
            print(f"❌ 未找到误差文件 (如 error_analysis_gpu.csv)")
            return
        
        print(f"正在加载误差报告: {os.path.basename(error_csv_path)}...")
        try:
            error_df = pd.read_csv(error_csv_path)
            # 兼容旧版误差报告 (如果没有 Bird_Index，默认全部为0)
            if 'Bird_Index' not in error_df.columns:
                error_df['Bird_Index'] = 0 
                # 尝试根据 True_Label 智能推断 (如果 labels.csv 存在)
                # 这里简单处理，假设单鸟
            
            work_items = error_df.to_dict('records')
            current_mode = "Correction"
        except Exception as e:
            print(f"读取 CSV 失败: {e}")
            return
        print(f"📉 共有 {len(work_items)} 个疑似误判样本等待复核。")
        
    # ================= 模式 1: 新图模式 =================
    else:
        print("扫描文件夹...")
        valid_exts = {'.nef', '.arw', '.cr2', '.cr3', '.jpg', '.jpeg', '.png'}
        all_files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
        
        # 读取已标记的 (Filename + Index) 集合
        labeled_keys = set()
        if os.path.exists(label_csv_path):
            try:
                existing_df = pd.read_csv(label_csv_path)
                # 兼容旧数据
                if 'Bird_Index' not in existing_df.columns: existing_df['Bird_Index'] = 0
                
                # 创建唯一键: "filename_index"
                for _, row in existing_df.iterrows():
                    key = f"{row['Filename']}_{int(row['Bird_Index'])}"
                    labeled_keys.add(key)
            except: pass
            
        # 我们先把所有文件加入列表，具体跳过逻辑在循环内判断（因为需要先识别出鸟的数量）
        # 这里为了 UI 进度条准确，简单地全量加入
        work_items = [{'Filename': f} for f in all_files]
        print(f"📸 文件夹内共有 {len(work_items)} 张图片。")

    if not work_items: return

    detector = BirdDetector()
    
    # === 加载主数据库 (Master DB) ===
    if os.path.exists(label_csv_path):
        master_df = pd.read_csv(label_csv_path)
        # 数据清洗与升级
        if 'Bird_Index' not in master_df.columns:
            print("⚠️ 升级数据库格式 (添加 Bird_Index)...")
            master_df['Bird_Index'] = 0
        if 'Label' in master_df.columns and 'Human_Label' not in master_df.columns:
            master_df.rename(columns={'Label': 'Human_Label'}, inplace=True)
    else:
        master_df = pd.DataFrame(columns=['Filename', 'Bird_Index', 'Human_Label'])

    # 确保 Bird_Index 是整数类型
    master_df['Bird_Index'] = master_df['Bird_Index'].fillna(0).astype(int)

    LABEL_MAP_REV = {0: 'NotBird', 1: 'Trash', 2: 'Soft', 3: 'Perfect'}
    
    print("\n--- 操作: [0]非鸟 [1]Trash [2]Soft [3]Perfect | [Space]跳过 [ESC]退出 ---")

    # ================= 主处理循环 =================
    for item in work_items:
        filename = item['Filename']
        path = os.path.join(folder, filename)
        
        # A. 如果是纠错模式，我们有目标 Bird_Index
        target_bird_idx = item.get('Bird_Index') # 可能为 None (新图模式)
        
        # B. 纠错模式的辅助信息
        ai_pred = item.get('Predicted_Label') or item.get('AI_Predict')
        old_label_str = item.get('True_Label')
        
        # 提取概率
        ai_conf = None
        if 'P_Trash' in item:
            try:
                probs = [float(item['P_Trash']), float(item['P_Soft']), float(item['P_Perfect'])]
                ai_conf = max(probs)
            except: pass

        # C. 加载图片
        full_img = load_best_available_image(path)
        if full_img is None: continue
        
        # D. 识别所有鸟
        rois = detector.detect_and_crop_all(full_img, standard_size=ROI_STANDARD_SIZE)
        if not rois: continue
        
        # ================= 遍历每一只鸟 =================
        for idx, roi in enumerate(rois):
            
            # 逻辑分支 1: 纠错模式 - 只显示指定的鸟
            if current_mode == "Correction":
                # 如果误差报告里有 index，严格匹配
                if target_bird_idx is not None:
                    if int(target_bird_idx) != idx: continue 
                else:
                    # 兼容旧版误差报告(无index): 默认只纠错第0只鸟(通常是最大的)
                    if idx != 0: continue

            # 逻辑分支 2: 新图模式 - 跳过已标记的鸟
            if current_mode == "New":
                unique_key = f"{filename}_{idx}"
                # 实时检查 Master DF (防止重复)
                is_labeled = ((master_df['Filename'] == filename) & (master_df['Bird_Index'] == idx)).any()
                if is_labeled: continue

            # E. 显示界面
            display_img = Visualizer.add_hud(
                roi, 
                filename, 
                bird_idx=idx,
                total_birds=len(rois),
                human_label=old_label_str, 
                ai_pred=ai_pred,
                ai_conf=ai_conf,
                mode=current_mode
            )
            
            cv2.imshow("Multi-Bird Labeler", display_img)
            
            # F. 等待交互
            new_label_code = -1
            while True:
                key = cv2.waitKey(0)
                if key == 27: # ESC
                    print("保存并退出...")
                    try:
                        master_df.to_csv(label_csv_path, index=False)
                    except PermissionError:
                        print("❌ 退出时保存失败：文件被占用！")
                    cv2.destroyAllWindows()
                    return
                elif key == 32: # Space
                    break
                elif key == ord('0'): new_label_code = 0; break # 非鸟
                elif key == ord('1'): new_label_code = 1; break
                elif key == ord('2'): new_label_code = 2; break
                elif key == ord('3'): new_label_code = 3; break
            
            # G. 更新数据
            if new_label_code != -1:
                # 查找是否已存在记录
                mask = (master_df['Filename'] == filename) & (master_df['Bird_Index'] == idx)
                
                if mask.any():
                    # 修正现有
                    old_val = master_df.loc[mask, 'Human_Label'].values[0]
                    master_df.loc[mask, 'Human_Label'] = new_label_code
                    action = f"修正 (原:{old_val})"
                else:
                    # 新增
                    new_row = pd.DataFrame([{
                        'Filename': filename, 
                        'Bird_Index': idx, 
                        'Human_Label': new_label_code
                    }])
                    master_df = pd.concat([master_df, new_row], ignore_index=True)
                    action = "新增"
                
                label_text = LABEL_MAP_REV.get(new_label_code, str(new_label_code))
                print(f"[{action}] {filename} (Bird {idx}) -> {label_text}")
                
                # 实时保存
                try:
                    master_df.to_csv(label_csv_path, index=False)
                except PermissionError:
                    print("❌ 保存失败：labels.csv 被占用！请关闭 Excel。")

    cv2.destroyAllWindows()
    print("当前列表处理完成！")

if __name__ == "__main__":
    main()