"""
3D 打印机喷头检测项目 - YOLOv8 训练与验证入口。

本模块只处理训练和验证；数据准备、去重、分析等步骤不在 v8 入口中重复实现。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .evaluation.error_sample_exporter import ErrorSampleExporter
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
    val.add_argument("--task", choices=("train", "val", "test"), default="val", help="验证 train、val 或 test 划分")
    val.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    val.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    val.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    val.add_argument("--project", default="runs/detect/val", help="验证输出目录")
    val.add_argument("--name", default="nozzle_ng_ok_v8", help="验证实验名称")
    val.add_argument("--save-txt", action="store_true", help="保存预测框 txt，错误样本复查建议开启")
    val.add_argument("--save-conf", action="store_true", help="在预测 txt 中保存置信度，需要配合 --save-txt 使用")
    val.add_argument("--save-json", action="store_true", help="保存 JSON 格式预测结果")
    val.add_argument("--export-error-samples", action="store_true", help="验证结束后按漏检、误检、类别错误复制错误样本到 runs 目录")
    val.add_argument("--error-samples-dir", default=None, help="错误样本输出根目录；不填时默认 runs/<task>/BadCase")
    val.add_argument("--error-iou-thres", type=float, default=0.5, help="错误样本匹配 IoU 阈值，默认0.5")
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
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_json=args.save_json,
        )
        result = runner.val(kwargs, dry_run=args.dry_run)
        print(result.summary)
        if result.returncode == 0 and args.export_error_samples and not args.dry_run:
            if not args.save_txt:
                print("未开启 --save-txt，无法导出错误样本；请开启 save_txt 后重新验证。")
                return result.returncode
            val_save_dir = _resolve_val_save_dir(result.payload, project=args.project, name=args.name)
            error_samples_dir = Path(args.error_samples_dir or f"runs/{args.task}/BadCase")
            summary = ErrorSampleExporter(
                data_yaml=Path(args.data),
                task=args.task,
                val_save_dir=val_save_dir,
                output_root=error_samples_dir,
                iou_threshold=args.error_iou_thres,
            ).export()
            print(f"错误样本已导出：{_build_error_output_dir(error_samples_dir, val_save_dir)}")
            print(
                f"漏检：{summary['missed_target']}，"
                f"误检：{summary['false_alarm']}，"
                f"类别错误：{summary['class_error']}，"
                f"涉及图片：{summary['images_with_errors']}"
            )
        return result.returncode

    parser.error(f"不支持的命令：{args.command}")
    return 2


def _resolve_val_save_dir(payload: object, project: str | Path, name: str) -> Path:
    """优先使用 Ultralytics 返回的 save_dir，缺失时回退到 project/name。"""
    save_dir = getattr(payload, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return Path(project) / name


def _build_error_output_dir(output_root: Path, val_save_dir: Path) -> Path:
    """构造人工复查输出目录提示。"""
    parts = val_save_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "val":
            return output_root / Path(*parts[index + 1 :])
    return output_root / val_save_dir.name


if __name__ == "__main__":
    raise SystemExit(main())
