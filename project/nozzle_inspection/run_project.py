"""YOLOv8 喷头检测项目的最短脚本入口。"""

from __future__ import annotations

from pathlib import Path

from .evaluation.error_sample_exporter import ErrorSampleExporter
from .run_config import CONFIG, RunConfig


AUGMENTATION_OFF_OVERRIDES = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


def run(config: RunConfig = CONFIG) -> int:
    """直接根据配置调用 Ultralytics，不再经过 CLI、main 或 Factory。"""
    if config.action == "train":
        return _train(config)
    if config.action == "val":
        return _val(config)
    raise ValueError("CONFIG.action 只支持 train 或 val")


def _train(config: RunConfig) -> int:
    iou_type = config.iou_type.lower()
    if iou_type not in {"ciou", "siou"}:
        raise ValueError("iou_type 只支持 'ciou' 或 'siou'")
    kwargs: dict[str, object] = {
        "data": _path(config.data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "optimizer": config.optimizer,
        "iou_type": iou_type,
        "workers": config.workers,
        "device": _detect_device(),
        "project": _absolute_path(config.train_project),
        "name": config.train_name,
        "exist_ok": config.exist_ok,
    }
    # 如果指定了超参数配置文件，则使用项目自定义的超参数；否则使用 YOLOv8 官方默认增强
    if config.hyp_yaml is not None:
        kwargs["cfg"] = _path(config.hyp_yaml)
    if not config.enable_augmentation:
        kwargs.update(AUGMENTATION_OFF_OVERRIDES)
    if config.dry_run:
        print(f"dry-run train: model={config.model}, {kwargs}")
        return 0

    from ultralytics import YOLO

    print(f"训练损失配置：IoU={iou_type.upper()}")
    YOLO(config.model).train(**kwargs)
    return 0


def _val(config: RunConfig) -> int:
    kwargs: dict[str, object] = {
        "data": _path(config.data_yaml),
        "split": config.eval_task,
        "conf": config.conf,
        "iou": config.iou,
        "imgsz": config.imgsz,
        "device": _detect_device(),
        "project": _absolute_path(config.val_project),
        "name": config.val_name,
        "save_txt": config.save_txt,
        "save_conf": config.save_conf,
        "save_json": config.save_json,
    }
    if config.dry_run:
        print(f"dry-run val: weights={config.weights}, {kwargs}")
        return 0

    from ultralytics import YOLO

    metrics = YOLO(_path(config.weights)).val(**kwargs)
    if config.export_error_samples:
        if not config.save_txt:
            print("未开启 save_txt，跳过 BadCase 导出。")
            return 0
        save_dir = Path(metrics.save_dir)
        summary = ErrorSampleExporter(
            data_yaml=config.data_yaml,
            task=config.eval_task,
            val_save_dir=save_dir,
            output_root=config.error_samples_dir,
            iou_threshold=config.error_iou_threshold,
        ).export()
        print(f"BadCase 已导出：{config.error_samples_dir / save_dir.name}")
        print(
            f"漏检：{summary['missed_target']}，误检：{summary['false_alarm']}，"
            f"类别错误：{summary['class_error']}，涉及图片：{summary['images_with_errors']}"
        )
    return 0


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        print(f"检测到 GPU：{torch.cuda.get_device_name(0)}")
        return "0"
    print("未检测到 GPU，使用 CPU")
    return "cpu"


def _absolute_path(path: Path) -> str:
    return str((path if path.is_absolute() else Path.cwd() / path).resolve())


def _path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(run())
