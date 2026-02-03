# **🦅 Bird Photo AI Sorter v2.1.0**

**The Ultimate AI Culling Tool for Bird Photographers.**

**告别“数毛”焦虑，让 AI 帮你从数千张连拍中挑出最锐利的瞬间。**

## **📖 简介 / Introduction**

**Bird Photo AI Sorter** 是专为生态摄影师打造的智能化后期筛选工具。针对鸟类摄影中“高速连拍、景深极浅、背景杂乱”的痛点，它利用计算机视觉与机器学习技术，实现像素级的锐度分析与自动化分类。

**v2.1.0 版本里程碑 (Milestone)**：

引入了 **"Dual-Engine" (双引擎)** 架构与 **"Mask-Aware V9"** 特征集。不仅解决了最新显卡（如 RTX 5060）的驱动兼容性问题，更通过引入高斯差分（DoG）和信噪比代理（SNR Proxy），显著提升了对高感光度（High ISO）噪点图的识别能力。

## **✨ 核心特性 / Key Features (v2.1.0)**

### **1\. 🚀 双引擎混合架构 (Dual-Engine Architecture)**

* **Auto-Switching**: 程序启动时自动检测硬件环境。  
  * **GPU 模式**: 检测到 NVIDIA 显卡时，利用 PyTorch CUDA 进行极速批量张量运算 (FFT/Sobel)。  
  * **CPU 模式**: 无显卡或驱动冲突时（如 Hyper-V 蓝屏问题），自动无缝切换至 OpenCV/NumPy 引擎，保证 **100% 稳定运行**。  
* **Consistency**: 严格对齐了 CPU 与 GPU 的数学逻辑，确保两套引擎输出的特征向量高度一致。

### **2\. 🎯 V9 掩膜感知特征 (Mask-Aware V9 Features)**

摒弃了传统的全局梯度评价，采用更符合物理光学的特征组合：

* **Difference of Gaussians (DoG)**: 模拟人眼视觉，通过带通滤波分离“真实纹理”与“随机噪点”。  
* **SNR Proxy**: 计算结构能量与高频杂讯的比率，有效剔除“假锐利”的噪点废片。  
* **Pixel-Level Masking**: 利用 **YOLOv8-Seg** 生成鸟类蒙版，并将背景强制涂黑，配合边缘二次腐蚀，彻底消除树枝背景干扰。

### **3\. 🧠 随机森林三级分类 (Smart Classification)**

基于 **Random Forest** 的非线性分类器，将照片分为三类：

* **🗑️ Trash (废片)**: 严重跑焦、模糊。  
* **😐 Soft (部分失焦)**: 缩图可用，但微观反差不足（缓冲区）。  
* **🏆 Perfect (完美)**: 核心区域（眼睛/羽毛）达到数毛级锐度。

### **4\. 🔄 人机协同数据闭环 (Human-in-the-loop)**

* **纠错模式**: 支持导出误判报告 (error\_analysis.csv)，并在标记工具中复盘 AI 的判断，修正后重新训练。这让模型能不断适应你的镜头特性和审美标准。

### **5\. 📥 Lightroom 无损工作流**

支持生成 .xmp 元数据文件。

* **Perfect** \-\> 🟢 绿色标签 \+ 5星  
* **Soft** \-\> 🔴 红色标签 \+ 3星  
* **Trash** \-\> 🟣 紫色标签 \+ 1星

## **📂 文件结构 / File Structure**

.  
├── batch\_processor\_locked\_dual.py     \# \[🚀 生产入口\] 双引擎主程序。自动选择 GPU/CPU 进行批量筛选。  
├── train\_classifier\_dual.py           \# \[🧠 训练入口\] 双引擎训练程序。支持自动参数寻优 (GridSearch)。  
├── data\_labler\_loop\_v2.py             \# \[🏷️ 标记工具\] 支持"新图标记"和"误判纠错"双模式。  
│  
├── feature\_extractor\_dual.py          \# \[⚙️ 核心算法\] 封装了 V9 特征提取逻辑 (DoG, SNR, Multi-Scale)。  
├── detect\_birds\_multi\_maskenabled.py  \# \[👁️ 视觉模型\] YOLOv8-Seg 实例分割与蒙版生成。  
├── metadata\_utils.py                  \# \[ℹ️ 元数据\] 读取 ISO、快门等 EXIF 信息。  
│  
├── best\_bird\_model\_multiclass.pkl     \# \[💾 权重文件\] 训练好的随机森林模型。  
├── yolov8x-seg.pt                     \# \[💾 视觉权重\] YOLO 分割模型。  
└── README.md

## **🚀 快速开始 / Quick Start**

### **1\. 安装依赖 (Installation)**

**基础依赖:**

pip install ultralytics opencv-python numpy scikit-learn tqdm rawpy joblib scipy exifread

**GPU 加速支持 (NVIDIA 显卡用户推荐):**

请前往 [PyTorch 官网](https://pytorch.org/) 复制适合您 CUDA 版本的安装命令。

*(注: RTX 40/50 系列用户建议安装 Nightly 版本以获得最佳兼容性)*

\# 示例 (CUDA 12.x)  
pip install \--pre torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/nightly/cu126\](https://download.pytorch.org/whl/nightly/cu126)

### **2\. 标准工作流 (Workflow)**

#### **第一步：建立基准 (Labeling)**

训练一个懂你的 AI。

1. 运行 python data\_labler\_loop\_v2.py。  
2. 选择 **\[1\] 新图标记**，对约 100-200 张照片进行打分 (1/2/3键)。

#### **第二步：训练模型 (Training)**

1. 运行 python train\_classifier\_dual.py。  
2. 程序会自动调用双引擎提取特征，并进行网格搜索寻找最佳参数。  
3. 训练完成后会生成 best\_bird\_model\_multiclass.pkl 和误判报告。

#### **第三步：批量筛选 (Sorting)**

1. 运行 python batch\_processor\_locked\_dual.py。  
2. 选择模式：  
   * **\[1\] 整理模式**：物理移动文件。  
   * **\[2\] 标注模式 (推荐)**：生成 XMP 文件。

## **📊 性能指标 (Benchmark)**

基于 v2.1.0 版本在 RTX 5060 Laptop 上的测试数据 (样本量 N=2000)：

| 类别 | Precision (精确率) | Recall (召回率) | 说明 |
| :---- | :---- | :---- | :---- |
| **Trash** | **86%** | **86%** | 极其精准，放心删除。 |
| **Perfect** | **88%** | **72%** | 只要选出来就是极品，宁缺毋滥。 |
| **Soft** | **76%** | **83%** | 充当安全缓冲区，防止好片被误删。 |

* **推理速度 (GPU)**: \~0.05秒 / 张 (批量模式)  
* **推理速度 (CPU)**: \~0.8秒 / 张 (多核并行)

## **🗺️ 路线图 / Roadmap**

* \[x\] **v1.0**: 基础功能，单线程 CPU。  
* \[x\] **v2.0**: 引入 GPU 加速，多目标识别，ISO 感知。  
* \[x\] **v2.1 (Current)**: 双引擎架构，DoG 降噪特征，Mask-Aware 像素级背景剔除。  
* \[ \] **v3.0**: 图形化界面 (GUI)，支持拖拽操作与实时预览。  
* \[ \] **Future**: 引入时序分析，针对连拍组进行“最佳瞬间”择优 (Best-of-Burst)。

## **🤝 贡献 / Contribution**

本项目开源于 GitHub: [Tony-Liu217/Bird-Photo-AI-Sorter](https://www.google.com/search?q=https://github.com/Tony-Liu217/Bird-Photo-AI-Sorter)

欢迎提交 Issue 反馈 Bug，或提交 Pull Request 贡献代码！