if __package__ in (None, ""):
    import sys
    from pathlib import Path

    # 兼容用户直接运行本文件的方式：python project/nozzle_inspection/run_project.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from project.nozzle_inspection.main import main
    from project.nozzle_inspection.run_config import CONFIG
else:
    from .main import main
    from .run_config import CONFIG


def _run_main(argv: list[object]) -> int:
    """argparse 只接受字符串参数，这里统一收口 Path 等配置对象。"""
    return main([str(arg) for arg in argv])


def run() -> int:
    """根据 run_config.py 中的 CONFIG 运行固定入口。"""
    if CONFIG.action == "train":
        argv = [
            "train",
            "--data",
            str(CONFIG.data_yaml),
            "--model",
            CONFIG.model,
            "--epochs",
            str(CONFIG.epochs),
            "--imgsz",
            str(CONFIG.imgsz),
            "--batch",
            str(CONFIG.batch),
            "--workers",
            str(CONFIG.workers),
            "--optimizer",
            CONFIG.optimizer,
            "--project",
            CONFIG.train_project,
            "--name",
            CONFIG.train_name,
        ]
        argv.append("--exist-ok" if CONFIG.exist_ok else "--no-exist-ok")
        argv.append("--enable-augmentation" if CONFIG.enable_augmentation else "--disable-augmentation")
        if CONFIG.dry_run:
            argv.append("--dry-run")
        return _run_main(argv)

    if CONFIG.action == "val":
        argv = [
            "val",
            "--data",
            str(CONFIG.data_yaml),
            "--weights",
            str(CONFIG.weights),
            "--task",
            CONFIG.eval_task,
            "--conf",
            str(CONFIG.conf),
            "--iou",
            str(CONFIG.iou),
            "--imgsz",
            str(CONFIG.imgsz),
            "--project",
            CONFIG.val_project,
            "--name",
            CONFIG.val_name,
        ]
        if CONFIG.dry_run:
            argv.append("--dry-run")
        return _run_main(argv)

    raise ValueError("CONFIG.action 只支持 train 或 val")


if __name__ == "__main__":
    raise SystemExit(run())
