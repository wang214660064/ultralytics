# 项目说明

本目录是 3D 打印机喷头/打印异常检测项目的 YOLOv8 子工程。所有 v8 相关训练、验证、测试和脚本运行，都应以本目录作为工作目录执行。根目录可能并列存在 `yolov5/`、`yolov11/` 等版本，不要跨版本混用路径、配置或权重。

## 沟通与代码风格

- 与用户的所有交流都使用中文。
- 新增或修改代码时，注释使用中文，并保证注释解释业务意图或关键约束，不写无意义注释。
- 优先保持现有项目结构和命名风格；不要为了小改动引入新的框架、目录层级或抽象。

## 运行环境

使用项目 conda 环境进行所有 Python 检查、脚本和测试。优先先激活 `yolov8` 环境，再直接运行 `python`，这样训练和验证日志可以实时输出：

```bash
conda activate yolov8
python ...
```

训练、验证等长时间任务优先使用非缓冲模式，便于观察 epoch、loss、mAP、验证指标等过程日志：

```bash
python -u ...
```

除非用户明确要求使用其他环境，否则不要使用系统 Python、基础 conda 环境、Codex 捆绑的 Python 运行时，或默认改回非实时输出的环境包装方式进行项目级测试。

## 工作目录要求

从根目录处理 YOLOv8 任务时，必须先进入本目录：

```bash
cd yolov8
conda activate yolov8
python ...
```

如果已经位于 `yolov8/` 目录内，则确认当前终端已激活 `yolov8` 环境后直接运行 `python ...`。不要在根目录直接运行 v8 脚本，除非脚本或用户明确要求这样做。原因是本工程中的默认路径、数据配置、训练输出和包导入都按 `yolov8/` 作为工作目录设计。

## 常见入口

```bash
python -u run_nozzle_project.py
python -m project.nozzle_inspection.main prepare-data
python -m project.nozzle_inspection.main analyze-data
python -m project.nozzle_inspection.main write-config
python -u -m project.nozzle_inspection.main train
python -u -m project.nozzle_inspection.main val --weights runs/train/nozzle_ng_ok7/weights/best.pt
```

## 项目流程特点

当前 v8 工程围绕喷头 NG/OK 检测流水线组织，包含数据准备、数据分析、配置生成、训练、验证和报告生成：

- 数据准备：去重、标签转换、训练/验证/测试集划分。
- 数据分析：统计类别、目标尺寸、样本分布并生成报告。
- 配置生成：写入 YOLO 数据配置。
- 训练验证：使用 YOLOv8 训练和评估喷头异常检测模型。
- 报告输出：生成 Markdown/PPTX 等实验材料。

处理任务时优先沿这些既有模块和入口定位问题，不要绕开项目封装直接堆临时脚本。

## 基本检查

进行轻量级语法验证：

```bash
python -m py_compile project/nozzle_inspection/main.py project/nozzle_inspection/run_config.py run_nozzle_project.py
```

进行依赖检查：

```bash
python -c "import torch, cv2, openpyxl; print(torch.__version__, cv2.__version__)"
```

如需跑测试：

```bash
python -m pytest tests/nozzle_inspection
```

## 文件编辑边界

- 除非用户要求更新生成文件，否则将生成的报告、训练输出、缓存和权重排除在源代码编辑之外。
- 不主动编辑 `runs/`、`project/nozzle_inspection/outputs/`、`*.cache`、训练生成图片、权重文件等生成物。
- 根目录下的数据集和参考资料默认只读；如需清洗、拆分或重建数据集，先说明影响范围。
- 优先编辑 `project/nozzle_inspection/` 下的项目封装代码；只有确有必要时才修改 YOLOv8 原始框架文件。

## 路径与平台约束

- 当前工作环境是 Windows PowerShell，路径中可能包含中文和空格，命令中需要注意引号和工作目录。
- 项目代码应尽量使用 `pathlib.Path` 处理路径，减少硬编码分隔符。
- Windows 下涉及 DataLoader 或子进程时，注意 `workers`、编码和启动方式的兼容性。

## 任务处理偏好

- 排查训练、验证、导入、路径问题时，优先沿实际调用链复核入口、配置、相对路径和工作目录。
- 对已有封装保持尊重：能通过 `run_config.py`、`run_nozzle_project.py` 或 `project.nozzle_inspection.main` 解决的，不新增平行入口。
- 如果用户要求"复核相关代码"，默认做代码路径级检查，并给出明确文件位置、根因和可验证命令。
