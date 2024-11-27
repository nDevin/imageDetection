# 🚀 YOLO实时目标检测项目

这是一个基于YOLO（You Only Look Once）的实时目标检测项目，可以处理视频和图片。本项目特别适合想要入门计算机视觉和深度学习的初学者。

## 🎯 主要功能

- 视频实时目标检测
- 支持多种YOLO模型（YOLOv8、YOLO11s）
- 可自定义置信度阈值
- 生成带检测框的输出视频
- 支持多种目标类别检测

## 📋 环境要求

开始使用前，请确保安装以下组件：

- Python 3.8 或更高版本
- OpenCV (cv2)
- Ultralytics YOLO

```bash
# 基础依赖
pip install torch torchvision
pip install transformers
pip install pillow
pip install opencv-python
pip install ultralytics
pip install tqdm
pip install ffmpeg-python

# 如果下载速度慢，可以使用清华源：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers pillow opencv-python ultralytics tqdm ffmpeg-python
```

### FFmpeg 安装

#### Windows:
1. 访问 https://www.ffmpeg.org/download.html
2. 下载 Windows 版本（选择 Windows builds）
3. 解压下载的文件到指定目录（如 `C:\ffmpeg`）
4. 添加环境变量：
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在"系统变量"中找到 Path
   - 点击"编辑" → "新建"
   - 添加 FFmpeg 的 bin 目录路径（如 `C:\ffmpeg\bin`）
5. 验证安装：打开新的命令行窗口，输入 `ffmpeg -version`

#### Linux:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Mac:
```bash
brew install ffmpeg
```

## 📁 项目结构

```
.
├── README.md
├── main.py
├── media/
│   ├── car.jpg
│   ├── cars.jpg
│   ├── cars.mp4
│   ├── dogs.jpg
│   ├── girls.jpg
│   ├── image.png
│   ├── output.gif
│   └── output.mp4
└── models/
    ├── coco.names
    ├── yolo11s.pt
    ├── yolov4.weights
    └── yolov8n.pt
```

## 🚀 快速开始

1. 克隆项目：
```bash
git clone https://gitcode.com/langgptai/ImageDetection.git
cd ImageDetection
```

2. 安装依赖：
```bash
pip install ultralytics opencv-python imageio
```

3. 运行检测脚本：
```bash
python main.py
```

4. 运行视频分析脚本
```
HF_ENDPOINT=https://hf-mirror.com python emotion.py
```
感谢：https://hf-mirror.com/ 提供的代理！

## 💡 工作原理

程序执行以下步骤：

1. 加载YOLO模型（默认使用yolo11s.pt）
2. 打开视频文件（cars.mp4）
3. 处理每一帧：
   - 检测目标
   - 绘制边界框
   - 添加标签和置信度得分
4. 将处理后的视频保存为output.mp4

## ⚙️ 配置说明

您可以在`main.py`中修改以下参数：

- 模型选择：将`yolo11s.pt`更改为其他模型，如`yolov8s.pt`
- 置信度阈值：修改`0.5`来调整检测灵敏度
- 输入/输出文件：根据需要更改视频文件名

## 🎯 支持检测的目标

模型可以检测多种目标，包括：
- 汽车
- 人物
- 动物
- 以及更多（取决于所使用的模型）

## 🔍 关键特性

- 实时目标检测
- 高精度识别
- 易于使用
- 可自定义配置
- 支持多种输入格式

## ⚠️ 常见问题解决

常见问题及解决方案：

1. **ModuleNotFoundError**：确保所有依赖都已正确安装
2. **CUDA/GPU错误**：检查GPU驱动和PyTorch安装情况
3. **找不到视频文件**：确认视频文件路径是否正确
4. **内存不足**：尝试使用更小的模型或降低视频分辨率

## 📧 技术支持

如有问题或建议，请在项目仓库中提交Issue。

## 🤝 代码说明

main.py的主要功能说明：

1. 导入必要的库
2. 加载YOLO模型
3. 打开视频文件
4. 设置输出视频参数
5. 逐帧处理视频
6. 在每一帧上绘制检测结果
7. 保存处理后的视频

## 🌟 使用技巧

1. 如果处理速度较慢，可以尝试：
   - 使用更轻量级的模型（如yolov8n.pt）
   - 降低视频分辨率
   - 使用GPU加速（如果可用）

2. 调整置信度阈值：
   - 提高阈值可减少误检
   - 降低阈值可提高检测率

## 🏷️ 关键词

目标检测、YOLO、计算机视觉、深度学习、Python、OpenCV、实时检测、视频处理、人工智能、机器学习