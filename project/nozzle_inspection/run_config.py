"""
YOLOv8 训练与验证运行配置。

用户通常只需要改这里的 CONFIG 字段，再从 yolov8 目录运行：
`python -u run_nozzle_project.py`

运行前请先在 PowerShell 中进入 yolov8 目录并激活环境：
`conda activate yolov8`

常见用法：
1. 训练模型：把 action 改为 "train"，按需调整 epochs、batch、enable_augmentation。
2. 验证模型：把 action 改为 "val"，把 weights 指向要评估的 best.pt。
   如果要复查训练集错误样本，把 eval_task 改为 "train"。
3. 只检查参数不真正运行：把 dry_run 改为 True，确认输出路径和参数后再改回 False。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """
    项目运行配置。

    action 可选：
    - train: 使用 Ultralytics YOLOv8 训练喷头 NG/OK 检测模型
    - val: 使用指定权重验证 train、val 或 test 划分

    路径说明：
    - 相对路径都以 yolov8 目录为基准。
    - 输出目录会在入口中转成绝对路径，避免落到 Ultralytics 全局 runs 目录。
    """

    # 运行模式。"train" 表示训练，"val" 表示验证。
    action: str = "val"

    # 数据配置文件。通常不用改；里面指向根目录下已经准备好的 out_cross_hash_clean 数据集。
    data_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")

    # 训练配置。
    # 预训练模型。轻量实验可用 yolov8n.pt；想提高精度可改为 yolov8s.pt、yolov8m.pt 等。
    model: str = "yolov8n.pt"
    # 训练轮数。快速调试可设小一些，正式训练再增大。
    epochs: int = 50
    # 输入图像尺寸。常用 640；显存不足时可降低。
    imgsz: int = 640
    # 批次大小。显存不足时优先调小这个值。
    batch: int = 32
    # DataLoader 工作进程数。Windows 下如果遇到多进程问题，可先改为 0。
    workers: int = 2
    # 优化器名称，直接传给 Ultralytics。
    optimizer: str = "AdamW"
    # 训练输出根目录。最终目录为 train_project / train_name。
    train_project: Path = Path(r"runs/train")
    # 训练实验名。权重默认会保存在 train_project / train_name / weights / best.pt。
    train_name: str = "nozzle_ng_ok_v8-训练有增强"
    # True 表示允许复用已有输出目录；False 表示目录存在时自动递增新目录。
    exist_ok: bool = False
    # 数据增强总开关。False 会显式关闭 mosaic、翻转、HSV 等增强；True 使用 Ultralytics 默认增强。
    enable_augmentation: bool = False
    # 试运行开关。True 只打印参数，不真正训练或验证。
    dry_run: bool = False

    # 验证配置。
    # 待验证权重路径。训练完成后通常指向 runs/detect/.../weights/best.pt。
    weights: Path = Path("runs/train/"+train_name+"/weights/best.pt")
    # 置信度阈值。想观察更严格的检测效果可调高，例如 0.5 或 0.7。
    conf: float = 0.5
    # NMS IoU 阈值。一般保持默认即可。
    iou: float = 0.45
    # 验证数据划分。"train" 复查训练集，"val" 评估验证集，"test" 评估测试集。
    eval_task: str = "train"
    # 验证输出根目录。最终目录为 val_project / val_name。
    val_project: Path = Path("runs/val/"+train_name)
    # 验证实验名。这里把 conf 拼进名字，方便比较不同置信度阈值。
    val_name: str = "exp-conf-"+ str(conf)
    # 保存 Ultralytics 原生预测结果，错误样本复查需要 save_txt=True。
    save_txt: bool = True
    save_conf: bool = True
    save_json: bool = True
    # 导出错误样本到 runs/<eval_task>/BadCase，便于人工复查和后续报告整理。
    export_error_samples: bool = True
    error_samples_dir: Path = Path("runs/"+eval_task+"/BadCase")
    error_iou_threshold: float = 0.5

CONFIG = RunConfig()
