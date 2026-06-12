"""
3D 打印机喷头检测项目 - YOLOv8 训练与验证入口。

本模块只处理训练和验证；数据准备、去重、分析等步骤不在 v8 入口中重复实现。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .factories.evaluator_factory import EvaluatorFactory
from .factories.trainer_factory import TrainerFactory
from .yolo_runner import UltralyticsRunner


# Windows 控制台和子进程统一使用 UTF-8，减少中文路径和日志乱码。
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


def detect_device() -> str:
    """自动选择训练/验证设备：优先第一张 GPU，否则使用 CPU。"""
    try:
        import torch
    except ImportError:
        print("PyTorch 未安装，使用 CPU")
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"检测到 GPU: {device_name}")
        return "0"

    print("未检测到 GPU，使用 CPU")
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D打印机喷头检测 YOLOv8 训练与验证入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="使用 Ultralytics YOLOv8 训练模型")
    train.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", help="YOLO 数据配置文件")
    train.add_argument("--model", default="yolov8n.pt", help="预训练模型或模型配置")
    train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    train.add_argument("--batch", type=int, default=8, help="批次大小")
    train.add_argument("--workers", type=int, default=2, help="DataLoader 工作进程数")
    train.add_argument("--optimizer", default="AdamW", help="优化器名称")
    train.add_argument("--project", default="runs/detect", help="训练输出目录")
    train.add_argument("--name", default="nozzle_ng_ok_v8", help="实验名称")
    train.add_argument("--exist-ok", dest="exist_ok", action="store_true", default=True, help="允许复用已有输出目录")
    train.add_argument("--no-exist-ok", dest="exist_ok", action="store_false", help="已有输出目录时自动递增新目录")
    train.add_argument("--enable-augmentation", dest="enable_augmentation", action="store_true", default=False, help="开启 Ultralytics 默认数据增强")
    train.add_argument("--disable-augmentation", dest="enable_augmentation", action="store_false", help="关闭训练数据增强")
    train.add_argument("--dry-run", action="store_true", help="只打印参数，不实际训练")

    val = subparsers.add_parser("val", help="验证训练后的 YOLOv8 权重")
    val.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", help="YOLO 数据配置文件")
    val.add_argument("--weights", required=True, help="待验证权重路径")
    val.add_argument("--task", choices=("val", "test"), default="val", help="验证 val 或 test 划分")
    val.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    val.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    val.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    val.add_argument("--project", default="runs/detect/val", help="验证输出目录")
    val.add_argument("--name", default="nozzle_ng_ok_v8", help="验证实验名称")
    val.add_argument("--dry-run", action="store_true", help="只打印参数，不实际验证")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = detect_device()
    runner = UltralyticsRunner()

    if args.command == "train":
        kwargs = TrainerFactory(
            model=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            device=device,
        ).build_train_kwargs(
            data_yaml=Path(args.data),
            epochs=args.epochs,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            enable_augmentation=args.enable_augmentation,
        )
        result = runner.train(kwargs, dry_run=args.dry_run)
        print(result.summary)
        return result.returncode

    if args.command == "val":
        kwargs = EvaluatorFactory(
            imgsz=args.imgsz,
            iou=args.iou,
            device=device,
        ).build_val_kwargs(
            data_yaml=Path(args.data),
            weights=Path(args.weights),
            conf=args.conf,
            task=args.task,
            project=args.project,
            name=args.name,
        )
        result = runner.val(kwargs, dry_run=args.dry_run)
        print(result.summary)
        return result.returncode

    parser.error(f"不支持的命令：{args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
