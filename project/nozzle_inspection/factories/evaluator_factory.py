from dataclasses import dataclass
from pathlib import Path


def _as_posix_path(path: Path) -> str:
    """统一路径分隔符，方便 Ultralytics 在 Windows 下记录和解析。"""
    return str(path).replace("\\", "/")


def _resolve_project_path(project: str | Path) -> str:
    """将验证输出目录固定到当前工程，避免落入 Ultralytics 全局 RUNS_DIR。"""
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = Path.cwd() / project_path
    return str(project_path.resolve())


@dataclass(frozen=True)
class EvaluatorFactory:
    """构建 Ultralytics YOLOv8 验证参数。"""

    imgsz: int = 640
    iou: float = 0.45
    device: str = "cpu"

    def build_val_kwargs(
        self,
        data_yaml: Path,
        weights: Path,
        conf: float = 0.25,
        task: str = "val",
        project: str | Path = "runs/detect",
        name: str = "result-2-val",
        save_txt: bool = False,
        save_conf: bool = False,
        save_json: bool = False,
    ) -> dict[str, object]:
        """返回可直接传给 YOLO(...).val() 的参数字典。"""
        return {
            "model": _as_posix_path(weights),
            "data": _as_posix_path(data_yaml),
            "split": task,
            "conf": conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "project": _resolve_project_path(project),
            "name": name,
            "save_txt": save_txt,
            "save_conf": save_conf,
            "save_json": save_json,
        }
