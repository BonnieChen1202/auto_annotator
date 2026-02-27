# AI-Auto-Annotator (智能目标检测自动标注工具)

一款轻量级的“人机协同”目标检测数据集标注工具。结合了深度学习模型的强大预测能力与 Web 前端灵活的交互能力，旨在成倍提升机器视觉数据集的标注效率。

本工具为前后端分离架构，后端采用 FastAPI 驱动 YOLOv8 提供实时推理，前端采用 HTML5 + Fabric.js 提供可交互的 Canvas 画板。

## ✨ 核心功能迭代记录

* **V1.0：基础推理 API**
    * 搭建 Python 虚拟环境，集成 `FastAPI` 与 `Ultralytics`。
    * 实现 `/predict` 接口，接收图片流并返回 YOLOv8 的 Bounding Box 绝对坐标与置信度。
* **V1.5：前端可视化与跨域打通**
    * 配置 FastAPI 的 `CORSMiddleware` 解决前后端跨域访问限制。
    * 引入 `Fabric.js`，实现图片上传后在网页端自动绘制 AI 预测的标注框。
* **V2.0：人机协同交互与 YOLO 导出**
    * **编辑与删除**：支持鼠标拖拽修改 AI 生成的框，支持选中后按 `Delete/Backspace` 一键删除误检框。
    * **标准格式导出**：内置坐标归一化转换算法，一键将画布上的框导出为标准的 YOLO `.txt` 训练集格式。
* **V2.1：自定义类别与困难样本收集**
    * 新增“手动绘制模式”，支持用户用鼠标手动框出 AI 漏检的物体。
    * 支持为手动绘制的新物体自定义 `Class ID` 和 `Class Name`，完善边缘情况（Hard Examples）的数据回流。

## 🛠️ 技术栈

* **后端**：Python 3.9, FastAPI, Uvicorn, Ultralytics (YOLOv8)
* **前端**：HTML5, CSS3, JavaScript (ES6+), Fabric.js (Canvas 渲染)
* **数学原理**：导出数据时，使用以下公式将绝对坐标转换为 YOLO 归一化坐标：
    $$x_{center} = \frac{x_{min} + x_{max}}{2 \times image\_width}$$
    $$y_{center} = \frac{y_{min} + y_{max}}{2 \times image\_height}$$
    $$w_{norm} = \frac{x_{max} - x_{min}}{image\_width}$$
    $$h_{norm} = \frac{y_{max} - y_{min}}{image\_height}$$

## 🚀 快速启动

### 1. 环境准备
建议使用 Conda 创建独立的虚拟环境：
```bash
conda create -n auto_label python=3.9
conda activate auto_label
```
### 2. 安装依赖
由于新版 NumPy 可能与底层 PyTorch 存在兼容性问题，建议锁定 NumPy 1.x 版本（已使用国内镜像源加速）：
```bash
pip install fastapi uvicorn python-multipart ultralytics -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
pip install "numpy<2.0" -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
### 3. 运行后端服务
在项目根目录下，启动 FastAPI 服务器：
```bash
python main.py
```
启动成功后，终端会显示 Uvicorn running on http://127.0.0.1:8000。首次运行会自动下载预训练的 yolov8n.pt 模型权重文件。
### 4. 运行前端界面
直接在操作系统的文件管理器中找到 index.html 文件，双击并在现代浏览器（推荐 Chrome、Edge 或 Safari）中打开，即可开始使用完整的大屏可视化标注功能。

## 💡 交互操作指南与快捷键
🤖 智能识别：点击“选择文件”上传图片后，系统会自动调用后端 YOLOv8 接口，并在前端画板上渲染出带类别和置信度的橙色边界框。

🖱️ 调整修改：鼠标点击选中任意检测框，可直接拖拽边缘的蓝色控制点改变长宽，或拖拽框的中心区域移动位置。

❌ 一键删除：选中不需要的错误检测框后，按下键盘上的 Delete 或 Backspace 键即可连同文字标签一起彻底移除。

✏️ 手动绘制 (收集漏检样本)：

在网页顶部的工具栏输入你想标记的新物体的 ID（如 1）和 名称（如 cat）。

点击蓝色的 “✏️ 开启手动绘制” 按钮（按钮将变红，鼠标指针变为十字准星）。

在图片上按住鼠标左键并拖拽，即可画出绿色的手工标注框。

绘制完成后，再次点击该按钮退出绘制模式。

⬇️ 导出格式：修改完毕后，点击 “⬇️ 导出 YOLO 格式 (.txt)” 按钮，工具会触发浏览器下载一个与当前图片同名的 .txt 训练标签文件。

## 问题
API 返回 500 内部错误 (RuntimeError: Numpy is not available)

原因：近期发布的 NumPy 2.0 大版本与当前许多 PyTorch（YOLO 的底层依赖）版本不兼容，导致张量 (Tensor) 计算环境无法初始化。

解决方案：卸载有冲突的 numpy，强制安装低于 2.0 的稳定版本。
```bash
pip uninstall numpy
pip install "numpy<2.0" -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
