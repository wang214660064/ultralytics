from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class YoloResult:
    """记录一次训练或验证调用的结果。"""

    action: str
    returncode: int
    summary: str
    payload: Any


class UltralyticsRunner:
    """通过 Python API 调用 Ultralytics，避免依赖外部 yolo 命令是否可用。"""

    def train(self, kwargs: dict[str, object], dry_run: bool = False) -> YoloResult:
        if dry_run:
            return YoloResult("train", 0, self._format_dry_run("train", kwargs), dict(kwargs))

        model_name = str(kwargs["model"])
        train_kwargs = {key: value for key, value in kwargs.items() if key != "model"}

        from ultralytics import YOLO

        metrics = YOLO(model_name).train(**train_kwargs)
        return YoloResult("train", 0, "train finished", metrics)

    def val(self, kwargs: dict[str, object], dry_run: bool = False) -> YoloResult:
        if dry_run:
            return YoloResult("val", 0, self._format_dry_run("val", kwargs), dict(kwargs))

        model_name = str(kwargs["model"])
        val_kwargs = {key: value for key, value in kwargs.items() if key != "model"}

        from ultralytics import YOLO

        metrics = YOLO(model_name).val(**val_kwargs)
        return YoloResult("val", 0, "val finished", metrics)

    @staticmethod
    def _format_dry_run(action: str, kwargs: dict[str, object]) -> str:
        """生成便于复现实验的 dry-run 摘要。"""
        parts = [f"{key}={value}" for key, value in sorted(kwargs.items())]
        return f"dry-run {action}: " + " ".join(parts)
