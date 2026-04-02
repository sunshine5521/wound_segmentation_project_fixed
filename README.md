# 小鼠创面智能分割系统

基于 EfficientNet-B3 Encoder + U-Net Decoder 的小鼠创面图像分割与面积分析平台，面向实验场景提供从单图智能诊断、人工精修、批量统计到训练监控的完整闭环。

项目当前以 Streamlit Web 应用为主要入口，强调“可视化、可修正、可追踪、可迭代”，适合用于创面图像的日常分析、数据整理与模型迭代优化。

## 项目亮点

- 支持单张创面图像的快速分割、面积计算与结果下载
- 支持实时拍照或本地上传，适配桌面端与移动端采图场景
- 支持掩膜手动修正、自动填孔、边缘微调与一键重置
- 支持批量图片分析并导出包含掩膜和 CSV 报告的 ZIP 数据包
- 支持训练日志、模型权重与关键配置的可视化监控
- 支持修正数据后直接重训，形成“分析 → 修正 → 训练 → 再分析”的闭环

## 功能概览

### 1. 首页概览

- 展示系统简介、核心能力与推荐使用流程
- 汇总主要模块入口，便于首次使用者快速理解系统

### 2. 智能诊断

- 支持本地上传图片或实时拍照
- 支持阈值调节、像素/毫米比例校准
- 支持边缘扩张/收缩与自动填充孔洞
- 支持原图、预测掩膜、叠加图和滑块对比
- 支持下载分割结果、叠加效果图
- 支持分析历史记录查看与 CSV 导出
- 推理过程中提供加载动画，减少等待不确定感

### 3. 数据集修正

- 支持对现有图片与掩膜进行逐张浏览
- 支持画笔涂抹、橡皮擦除、笔刷大小调整
- 支持边缘微调与自动填孔
- 支持修正结果实时预览
- 支持一键重置当前掩膜
- 支持保存修正结果并直接触发重新训练

### 4. 批量处理

- 支持多张图片批量上传和自动处理
- 展示处理进度与单图异常跳过提示
- 自动输出像素面积和实际面积统计表
- 一键下载完整 ZIP 数据包，内含全部掩膜图片与 `batch_results.csv`

### 5. 系统监控

- 展示训练日志曲线
- 查看当前训练参数与关键配置
- 浏览已有模型权重文件与 checkpoint

## 技术栈

- Python 3.8+
- PyTorch
- segmentation-models-pytorch
- OpenCV
- Albumentations
- Streamlit

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
streamlit run web_app.py
```

默认访问地址：

```text
http://localhost:8501
```

Windows 下也可以直接使用启动脚本：

```bash
start_web.bat
```

## 命令行用法

除 Web 页面外，项目还提供命令行入口：

```bash
python run.py train
python run.py predict --image path/to/image.jpg
python run.py batch --dir path/to/images
```

适用场景：

- `train`：启动模型训练
- `predict`：对单张图片执行预测
- `batch`：对目录中的多张图片执行批量分析

## 推荐使用流程

1. 在“智能诊断”页面上传图片或直接拍照，快速观察分割效果
2. 若边缘存在误差或反光空洞，在“数据集修正”页面进行人工精修
3. 保存修正后的数据并重新训练模型
4. 在“批量处理”页面完成整批实验图片统计
5. 在“系统监控”页面检查训练曲线、参数与模型权重

## 服务器部署说明

项目可部署在本地工作站或 GPU 服务器上运行。

### 本地部署

```bash
pip install -r requirements.txt
streamlit run web_app.py
```

### AutoDL / Linux 服务器部署

推荐使用固定端口启动，例如：

```bash
streamlit run web_app.py --server.port 6006
```

若希望关闭终端后服务仍持续运行，建议使用 `tmux`：

```bash
tmux new -s myserver
streamlit run web_app.py --server.port 6006
```

启动成功后按：

```text
Ctrl + B，然后按 D
```

即可将服务挂到后台。之后可通过：

```bash
tmux attach -t myserver
```

重新进入会话。

## 配置说明

项目主要配置位于 `config.yaml`，包括：

- 数据目录与标注文件路径
- 图像输入尺寸
- 默认设备（CPU / CUDA）
- 像素与毫米换算比例
- 训练轮次、学习率、batch size
- checkpoint 与日志目录

如果需要进行面积物理单位换算，请优先确认 `pixels_per_mm` 配置是否正确。

## 项目结构

```text
├── checkpoints/              # 模型权重与训练保存文件
├── data/                     # 数据目录
├── logs/                     # 训练日志
├── scripts/                  # 数据处理与辅助脚本
├── src/
│   ├── dataset.py            # 数据集定义
│   ├── inference.py          # 推理与可视化逻辑
│   ├── model.py              # 模型结构
│   ├── train.py              # 训练流程
│   └── utils.py              # 通用工具函数
├── config.yaml               # 全局配置文件
├── requirements.txt          # Python 依赖
├── run.py                    # 命令行入口
├── start_web.bat             # Windows 启动脚本
└── web_app.py                # Streamlit Web 主程序
```

## 当前版本更新摘要

相较于早期版本，当前项目已补充以下体验优化：

- 智能诊断支持“实时拍照”
- 推理过程增加加载动画
- 历史分析记录支持导出 CSV
- 批量处理支持一键下载完整结果 ZIP
- 数据修正页面支持一键重置掩膜
- 推理结果会自动缩放回原始图像尺寸，降低尺寸不一致带来的报错风险

## 注意事项

- 首次运行前请确认 `checkpoints/best_model.pth` 已准备完成
- 若使用 GPU，请确认 CUDA 环境与 PyTorch 版本兼容
- 若在服务器部署，请注意端口开放方式与进程后台保活
- 若修改了 `web_app.py` 或 `src/inference.py`，同步到服务器后需要重启服务
