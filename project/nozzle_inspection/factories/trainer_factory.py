from dataclasses import dataclass
from pathlib import Path


def _as_posix_path(path: Path) -> str:
    """统一路径分隔符，减少 Windows 命令和日志里的路径差异。"""
    return str(path).replace("\\", "/")


def _resolve_project_path(project: str | Path) -> str:
    """
    将输出目录解析为绝对路径。

    Ultralytics 会把相对 project 挂到全局 RUNS_DIR 下；这里提前转绝对路径，
    确保结果保存在当前 yolov8 子工程内。
    """
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = Path.cwd() / project_path
    return str(project_path.resolve())


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


@dataclass(frozen=True)
class TrainerFactory:
    """构建 Ultralytics YOLOv8 训练参数。"""

    model: str = "yolov8n.pt"
    imgsz: int = 640
    batch: int = 8
    optimizer: str = "AdamW"
    device: str = "cpu"

    def build_train_kwargs(
        self,
        data_yaml: Path,
        epochs: int,
        workers: int = 2,
        project: str | Path = "runs/detect",
        name: str = "result-2",
        exist_ok: bool = True,
        enable_augmentation: bool = False,
    ) -> dict[str, object]:
        """返回可直接传给 YOLO(...).train() 的参数字典。"""
        kwargs: dict[str, object] = {
            "model": self.model,
            "data": _as_posix_path(data_yaml),
            "epochs": epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "optimizer": self.optimizer,
            "workers": workers,
            "device": self.device,
            "project": _resolve_project_path(project),
            "name": name,
            "exist_ok": exist_ok,
        }
        if not enable_augmentation:
            kwargs.update(AUGMENTATION_OFF_OVERRIDES)
        return kwargs
