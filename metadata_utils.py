import exifread
import os

def get_iso_speed(file_path):
    """
    从图片 (RAW/JPG) 中读取 ISO 感光度。
    如果读取失败，返回默认值 800 (作为中位数)。
    """
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # 尝试不同的常见 ISO 标签键值
            iso_keys = ['EXIF ISOSpeedRatings', 'Image ISOSpeedRatings', 'EXIF ISO']
            
            for key in iso_keys:
                if key in tags:
                    # exifread 返回的是 IfdTag 对象，需要转字符串再转 int
                    return int(str(tags[key]))
                    
            # 某些厂商的特殊标记
            return 800
            
    except Exception as e:
        # print(f"ISO 读取失败 {os.path.basename(file_path)}: {e}")
        return 800

# 测试代码
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename()
    if path:
        print(f"ISO: {get_iso_speed(path)}")