# **🦅 鸟类摄影 AI 自动筛选工具 (Bird Photo AI Sorter) v1.0**

**告别“数毛”焦虑，让 AI 帮你从数千张连拍中挑出最锐利的瞬间。**

## **📖 项目简介 / About This Project**

对于生态摄影师（特别是“打鸟”爱好者）来说，按一次快门可能意味着数十张连拍。一次拍摄回来，面对硬盘里数千张 RAW 格式的鸟类照片，人工筛选合焦准确、羽毛清晰的“数毛片”是一项巨大且枯燥的工程。

**Bird Photo AI Sorter** 是一个基于 **深度学习 (Deep Learning)** 和 **计算机视觉 (Computer Vision)** 的自动化筛选工具。它能自动识别鸟类主体，物理剔除背景干扰，利用物理光学的频域分析算法评估锐度，并使用机器学习模型将照片自动分类。

## **✨ 核心特性 / Key Features**

### **1\. 🎯 像素级精准识别 (Mask-Aware)**

采用 **YOLOv8-Instance Segmentation** 模型，不仅能框出鸟，还能生成**像素级蒙版（Mask）**。系统会将非鸟类区域（如锐利的前景树叶、树枝）强制涂黑，并进行边缘腐蚀，彻底防止背景干扰导致的误判。

### **2\. ⚡ 全流程 GPU 加速**

基于 PyTorch Tensor Batch 架构构建。利用 NVIDIA 显卡（支持 RTX 40/50 系列）并行加载、解码并分析批量 RAW 格式照片。

* **特性**：动态半精度推理 (FP16) \+ 多线程 IO 读取，吞吐量极高。

### **3\. 🔬 科学的锐度分析**

摒弃传统的单一梯度评价，引入 **FFT（快速傅里叶变换）** 分析中高频纹理（羽毛）的能量，同时智能过滤高感光度带来的噪点。

* **内部梯度峰值**：只计算鸟类躯干内部最锐利的 5% 像素，区分“微糊”和“极致”。

### **4\. 🧠 智能三级分类**

基于 **Random Forest (随机森林)** 分类器，将照片智能分为：

* **🗑️ Trash (废片)**：严重跑焦，建议直接删除。  
* **😐 Soft (部分失焦)**：缩图可用，但细节不足，需人工复审。  
* **🏆 Perfect (完美)**：数毛级锐利，直接出片。

### **5\. 📥 无损工作流 (Lightroom Friendly)**

支持生成标准 .xmp 元数据文件。导入 Adobe Lightroom Classic 后自动标记颜色标签（紫/红/绿）和星级，**无需物理移动 RAW 文件**，保护原始资产。

## **📂 文件结构 / File Structure**

本项目核心代码由以下几个模块组成：

.  
├── batch\_processor\_GPU\_optimize.py    \# \[🚀 主程序\] 生产环境入口。运行此文件开始批量筛选照片。  
│                                      \# 包含全流程逻辑：加载 \-\> 识别 \-\> 特征提取 \-\> 分类 \-\> 归档/标注。  
│  
├── detect\_birds\_multi\_maskenabled.py  \# \[👁️ 视觉核心\] 封装了 YOLOv8 实例分割逻辑。  
│                                      \# 负责图像的智能裁切、Mask 生成以及背景像素剔除。  
│  
├── data\_labler.py                     \# \[🏷️ 标记工具\] 用于人工构建"真值数据"。  
│                                      \# 提供快捷键交互界面，帮助用户快速将照片标记为 Trash/Soft/Perfect。  
│  
├── train\_classifier\_GPU.py            \# \[🧠 训练中心\] 用于训练专属 AI 模型。  
│                                      \# 读取标记数据，利用 GPU 加速提取特征，并训练随机森林分类器。  
│  
├── best\_bird\_model\_multiclass.pkl     \# \[💾 模型文件\] 训练好的随机森林分类器权重。  
│                                      \# 主程序运行时会直接加载此文件进行推理。  
│  
├── .gitignore                         \# Git 忽略配置，防止上传临时文件或过大的 PyTorch 模型。  
└── README.md                          \# 项目说明文档。

## **🚀 快速开始 / Quick Start**

### **1\. 环境依赖 (Prerequisites)**

本项目严重依赖 GPU 计算，请确保安装了 NVIDIA 显卡驱动及 CUDA 支持。

**安装 PyTorch (针对 RTX 30/40/50 系列推荐 Nightly 版):**

pip install \--pre torch torchvision torchaudio \--index-url \[https://download.pytorch.org/whl/nightly/cu126\](https://download.pytorch.org/whl/nightly/cu126)

**安装其他依赖库:**

pip install ultralytics opencv-python numpy scikit-learn tqdm rawpy joblib

### **2\. 使用工作流 (Workflow)**

#### **第一步：建立标准 (Labeling)**

每个人的镜头素质和对“锐利”的定义不同，建议先训练专属模型。

1. 准备 100-200 张典型照片（包含清晰和模糊的）。  
2. 运行标记工具：  
   python data\_labler.py

3. 按键操作：1 (废片), 2 (一般), 3 (完美), ESC (保存退出)。这将生成 labels.csv。

#### **第二步：训练模型 (Training)**

利用标记好的数据训练 AI。

1. 运行训练脚本：  
   python train\_classifier\_GPU.py

2. 程序会自动进行参数寻优，并生成 best\_bird\_model\_multiclass.pkl。

#### **第三步：批量筛选 (Sorting)**

开始处理海量照片。

1. 运行主程序：  
   python batch\_processor\_GPU\_optimize.py

2. 选择模式：  
   * **\[1\] 整理模式**：将文件物理移动到 Trash, Soft, Perfect 文件夹。  
   * **\[2\] 标注模式 (推荐)**：生成 .xmp 文件，保持原文件不动。

## **📥 Lightroom 导入指南**

如果您选择了 **\[2\] 标注模式**，请按以下步骤操作：

1. 在 Lightroom Classic 中导入照片。  
2. 如果照片已在库中，选中照片 \-\> 右键 \-\> **元数据** \-\> **从文件读取元数据**。  
3. 查看结果：  
   * **🟢 绿色标签 (5星)**：Perfect (完美)  
   * **🔴 红色标签 (3星)**：Soft (待定)  
   * **🟣 紫色标签 (1星)**：Trash (排除)

## **🗺️ 未来优化方向 / Future Roadmap**

目前的 v1.0 版本已经具备生产力，但针对复杂的鸟类摄影场景，我们计划在后续版本中重点攻克以下问题：

### **1\. 优化 "Soft" 中间态的纯净度**

* **现状**：目前的 Soft (部分失焦) 组是一个“缓冲区”，其中混杂了画质尚可的片子，也混入了噪点过高或极其轻微跑焦的废片，导致人工复审工作量依然存在。  
* **计划**：引入更细粒度的特征分析（如色彩噪声分离），或尝试将三分类扩展为五分类（增加 ISO\_High 和 Motion\_Blur 标签），进一步提纯 Soft 组。

### **2\. 增强对“飞版”运动模糊的识别**

* **现状**：对于飞行中的鸟类（飞版），算法可能会因为“运动模糊（Motion Blur）”导致的边缘柔化，将其误判为失焦（Soft）甚至废片。但实际上，飞版照片通常允许主体有轻微的动态模糊。  
* **计划**：引入 **方向性梯度检测 (Directional Gradient)** 或光流法，识别线性的运动模糊，给予飞版照片特定的权重补偿，避免“误杀”精彩的动态瞬间。

### **3\. 提升硬件兼容性**

* **现状**：目前的算法深度绑定 NVIDIA CUDA 生态，严重依赖 GPU 进行 FFT 和 Tensor 运算，无法在 Mac (Metal) 或 AMD 显卡上高效运行。  
* **计划**：重构推理后端，增加对 ONNX Runtime 或 MPS (Mac) 加速的支持，降低使用门槛。

### **4\. 开发图形化界面 (GUI)**

* **现状**：目前主要通过命令行交互，对非技术用户不够友好。  
* **计划**：开发基于 PyQt 或 Gradio 的可视化操作界面，支持拖拽文件夹、实时预览 ROI 裁切效果、手动微调阈值以及直观的图表分析。

### **5\. 参数配置与调优流程优化**

* **现状**：核心参数（如 ROI 尺寸、FFT 截止频率）目前硬编码在 Python 文件中，调整不便。  
* **计划**：  
  * 引入 config.yaml 配置文件，实现参数与代码分离。  
  * 开发“一键校准向导”，让用户只需导入几张典型照片，系统自动推荐最佳参数配置。

## **📝 License**

MIT License. Designed for photographers, by photographers.