"""
错误样本导出模块 - 从 YOLOv8 验证结果中复制人工复查样本。

该模块读取 Ultralytics 验证输出的预测 txt，与数据集真实 label 做 IoU 匹配，
并把漏检、误检、类别错误样本复制到 runs 下，方便人工复查。
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class YoloBox:
    """单个 YOLO 格式框。"""

    cls: int
    x: float
    y: float
    w: float
    h: float
    conf: float | None = None

    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.w / 2
        half_h = self.h / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )


@dataclass(frozen=True)
class ErrorSampleExporter:
    """
    错误样本导出器。

    data_yaml: YOLO 数据集配置文件。
    task: train/val/test，对应要分析的数据划分。
    val_save_dir: Ultralytics val 输出目录，里面应包含 labels/*.txt。
    output_root: 错误样本输出根目录，默认建议放在 runs/error_samples。
    iou_threshold: 判断预测框和真实框是否匹配的 IoU 阈值。
    """

    data_yaml: Path
    task: str
    val_save_dir: Path
    output_root: Path = Path("runs/error_samples")
    iou_threshold: float = 0.5

    def export(self) -> dict[str, int]:
        data = _load_dataset_yaml(self.data_yaml)
        image_dir = _resolve_split_dir(data, self.task)
        label_dir = _image_dir_to_label_dir(image_dir)
        prediction_dir = self.val_save_dir / "labels"
        review_root = self.output_root / _review_subdir(self.val_save_dir)
        if review_root.exists():
            shutil.rmtree(review_root)
        review_root.mkdir(parents=True, exist_ok=True)

        summary = {
            "missed_target": 0,
            "false_alarm": 0,
            "class_error": 0,
            "images_with_errors": 0,
        }
        rows: list[dict[str, str]] = []

        for image_path in _iter_images(image_dir):
            stem = image_path.stem
            gt_path = label_dir / f"{stem}.txt"
            pred_path = prediction_dir / f"{stem}.txt"
            gt_boxes = _read_yolo_boxes(gt_path, has_conf=False)
            pred_boxes = _read_yolo_boxes(pred_path, has_conf=True)
            image_errors = _classify_errors(gt_boxes, pred_boxes, self.iou_threshold)

            if not image_errors:
                continue

            summary["images_with_errors"] += 1
            for error_type in image_errors:
                summary[error_type] += 1
                _copy_review_files(image_path, gt_path, pred_path, review_root / error_type)
                rows.append(
                    {
                        "image": str(image_path),
                        "label": str(gt_path),
                        "prediction": str(pred_path),
                        "error_type": error_type,
                    }
                )

        _write_summary_csv(review_root / "error_summary.csv", rows)
        _write_badcase_config(
            review_root / "badcase_config.yaml",
            data_yaml=self.data_yaml,
            task=self.task,
            val_save_dir=self.val_save_dir,
            output_root=self.output_root,
            image_dir=image_dir,
            label_dir=label_dir,
            prediction_dir=prediction_dir,
            iou_threshold=self.iou_threshold,
        )
        _write_badcase_summary(review_root / "badcase_summary.md", summary, rows)
        return summary


def _load_dataset_yaml(data_yaml: Path) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    path = Path(data.get("path") or ".")
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    data["path"] = path
    return data


def _resolve_split_dir(data: dict, task: str) -> Path:
    if task not in {"train", "val", "test"}:
        raise ValueError(f"不支持的评估划分：{task}")
    split = data.get(task)
    if not split:
        raise ValueError(f"dataset.yaml 中缺少 {task} 路径")
    if isinstance(split, list):
        split = split[0]
    split_path = Path(split)
    if not split_path.is_absolute():
        split_path = data["path"] / split_path
    return split_path.resolve()


def _review_subdir(val_save_dir: Path) -> Path:
    parts = val_save_dir.parts
    for index in range(len(parts) - 1):
        if parts[index] == "val":
            return Path(*parts[index + 1 :])
    return Path(val_save_dir.name)


def _image_dir_to_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def _iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _read_yolo_boxes(path: Path, has_conf: bool) -> list[YoloBox]:
    if not path.exists():
        return []

    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        conf = float(parts[5]) if has_conf and len(parts) >= 6 else None
        boxes.append(
            YoloBox(
                cls=int(float(parts[0])),
                x=float(parts[1]),
                y=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
                conf=conf,
            )
        )
    return boxes


def _classify_errors(gt_boxes: list[YoloBox], pred_boxes: list[YoloBox], iou_threshold: float) -> list[str]:
    errors: list[str] = []
    matched_predictions: set[int] = set()

    for gt in gt_boxes:
        best_index = None
        best_iou = 0.0
        for index, pred in enumerate(pred_boxes):
            if index in matched_predictions:
                continue
            iou = _box_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is None or best_iou < iou_threshold:
            errors.append("missed_target")
            continue

        matched_predictions.add(best_index)
        if pred_boxes[best_index].cls != gt.cls:
            errors.append("class_error")

    for index in range(len(pred_boxes)):
        if index not in matched_predictions:
            errors.append("false_alarm")

    return sorted(set(errors))


def _box_iou(first: YoloBox, second: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy()
    bx1, by1, bx2, by2 = second.xyxy()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _copy_review_files(image_path: Path, label_path: Path, pred_path: Path, target_dir: Path) -> None:
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)
    (target_dir / "predictions").mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, target_dir / "images" / image_path.name)
    if label_path.exists():
        shutil.copy2(label_path, target_dir / "labels" / label_path.name)
    if pred_path.exists():
        shutil.copy2(pred_path, target_dir / "predictions" / pred_path.name)


def _write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image", "label", "prediction", "error_type"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_badcase_config(
    path: Path,
    data_yaml: Path,
    task: str,
    val_save_dir: Path,
    output_root: Path,
    image_dir: Path,
    label_dir: Path,
    prediction_dir: Path,
    iou_threshold: float,
) -> None:
    config = {
        "data_yaml": str(data_yaml),
        "task": task,
        "val_save_dir": str(val_save_dir),
        "output_root": str(output_root),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "prediction_dir": str(prediction_dir),
        "iou_threshold": iou_threshold,
        "error_types": {
            "missed_target": "真实有目标但没有匹配预测框",
            "false_alarm": "预测框没有匹配真实目标",
            "class_error": "预测框匹配真实目标但类别错误",
        },
    }
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_badcase_summary(path: Path, summary: dict[str, int], rows: list[dict[str, str]]) -> None:
    lines = [
        "# BadCase 复查摘要",
        "",
        "## 统计结果",
        "",
        f"- 漏检：{summary['missed_target']}",
        f"- 误检：{summary['false_alarm']}",
        f"- 类别错误：{summary['class_error']}",
        f"- 涉及图片：{summary['images_with_errors']}",
        "",
        "## 输出说明",
        "",
        "- `error_summary.csv`：逐样本错误清单，可直接用于实验报告统计。",
        "- `missed_target/`：真实有目标但没有匹配预测框的样本。",
        "- `false_alarm/`：预测框没有匹配真实目标的样本。",
        "- `class_error/`：预测框匹配真实目标但类别错误的样本。",
        "- 每个错误目录下的 `images/`、`labels/`、`predictions/` 分别保存原图、真实标签和预测标签。",
        "",
        "## 复查清单",
        "",
    ]
    if rows:
        lines.append("| 错误类型 | 图片 |")
        lines.append("| --- | --- |")
        for row in rows[:200]:
            lines.append(f"| {row['error_type']} | {Path(row['image']).name} |")
    else:
        lines.append("本次未导出 BadCase 样本。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
