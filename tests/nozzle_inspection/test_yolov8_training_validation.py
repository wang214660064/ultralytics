import unittest
import runpy
from pathlib import Path
from unittest.mock import patch

import project.nozzle_inspection.run_project as run_project
from project.nozzle_inspection.factories.evaluator_factory import EvaluatorFactory
from project.nozzle_inspection.factories.trainer_factory import TrainerFactory
from project.nozzle_inspection.main import main
from project.nozzle_inspection.run_config import RunConfig
from project.nozzle_inspection.yolo_runner import YoloResult


class Yolov8TrainingValidationTest(unittest.TestCase):
    def test_run_project_can_be_loaded_as_direct_script(self):
        namespace = runpy.run_path(
            "project/nozzle_inspection/run_project.py",
            run_name="nozzle_direct_script_test",
        )

        self.assertIn("run", namespace)

    def test_run_project_val_passes_only_string_arguments_to_main(self):
        config = RunConfig(
            action="val",
            dry_run=True,
            val_project=Path("runs/detect"),
            weights=Path("runs/detect/result-2/weights/best.pt"),
        )

        with patch.object(run_project, "CONFIG", config), \
             patch.object(run_project, "main", return_value=0) as main_func:
            returncode = run_project.run()

        self.assertEqual(returncode, 0)
        argv = main_func.call_args.args[0]
        self.assertTrue(all(isinstance(arg, str) for arg in argv))

    def test_default_badcase_output_uses_task_specific_badcase_folder(self):
        config = RunConfig()

        self.assertEqual(config.error_samples_dir, Path("runs/train/BadCase"))

    def test_trainer_builds_ultralytics_train_kwargs(self):
        kwargs = TrainerFactory(device="0").build_train_kwargs(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            epochs=3,
            workers=0,
            enable_augmentation=True,
        )

        self.assertEqual(kwargs["model"], "yolov8n.pt")
        self.assertEqual(kwargs["data"], "project/nozzle_inspection/configs/dataset.yaml")
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(kwargs["optimizer"], "AdamW")
        self.assertEqual(kwargs["device"], "0")
        self.assertEqual(kwargs["workers"], 0)
        self.assertEqual(kwargs["project"], str((Path.cwd() / "runs" / "detect").resolve()))
        self.assertEqual(kwargs["name"], "result-2")
        self.assertTrue(kwargs["exist_ok"])
        self.assertNotIn("mosaic", kwargs)

    def test_trainer_disables_detection_augmentation_by_default(self):
        kwargs = TrainerFactory(device="0").build_train_kwargs(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            epochs=3,
            workers=0,
        )

        for key in (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
        ):
            self.assertEqual(kwargs[key], 0.0)

    def test_trainer_absolutizes_custom_relative_project(self):
        kwargs = TrainerFactory(device="0").build_train_kwargs(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            epochs=3,
            workers=0,
            project="custom/runs",
            name="exp-a",
        )

        self.assertEqual(kwargs["project"], str((Path.cwd() / "custom" / "runs").resolve()))
        self.assertEqual(kwargs["name"], "exp-a")

    def test_evaluator_builds_ultralytics_val_kwargs_for_test_split(self):
        kwargs = EvaluatorFactory(device="cpu").build_val_kwargs(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/nozzle_ng_ok_v8/weights/best.pt"),
            conf=0.7,
            task="test",
        )

        self.assertEqual(kwargs["model"], "runs/train/nozzle_ng_ok_v8/weights/best.pt")
        self.assertEqual(kwargs["data"], "project/nozzle_inspection/configs/dataset.yaml")
        self.assertEqual(kwargs["split"], "test")
        self.assertEqual(kwargs["conf"], 0.7)
        self.assertEqual(kwargs["device"], "cpu")

    def test_evaluator_can_save_predictions_for_error_review(self):
        kwargs = EvaluatorFactory(device="cpu").build_val_kwargs(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/nozzle_ng_ok_v8/weights/best.pt"),
            conf=0.25,
            save_txt=True,
            save_conf=True,
            save_json=True,
        )

        self.assertTrue(kwargs["save_txt"])
        self.assertTrue(kwargs["save_conf"])
        self.assertTrue(kwargs["save_json"])

    def test_main_train_uses_runner_with_dry_run(self):
        result = YoloResult(
            action="train",
            returncode=0,
            summary="dry-run train",
            payload={"epochs": 1},
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.UltralyticsRunner") as runner_cls:
            runner_cls.return_value.train.return_value = result

            returncode = main(["train", "--epochs", "1", "--workers", "0", "--dry-run"])

        self.assertEqual(returncode, 0)
        runner_cls.return_value.train.assert_called_once()
        kwargs = runner_cls.return_value.train.call_args.args[0]
        self.assertEqual(kwargs["epochs"], 1)
        self.assertEqual(kwargs["workers"], 0)
        self.assertEqual(kwargs["mosaic"], 0.0)

    def test_main_train_can_enable_augmentation(self):
        result = YoloResult(
            action="train",
            returncode=0,
            summary="dry-run train",
            payload={"epochs": 1},
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.UltralyticsRunner") as runner_cls:
            runner_cls.return_value.train.return_value = result

            returncode = main(["train", "--epochs", "1", "--workers", "0", "--enable-augmentation", "--dry-run"])

        self.assertEqual(returncode, 0)
        kwargs = runner_cls.return_value.train.call_args.args[0]
        self.assertNotIn("mosaic", kwargs)

    def test_main_val_passes_requested_split_and_confidence(self):
        result = YoloResult(
            action="val",
            returncode=0,
            summary="dry-run val",
            payload={"split": "test"},
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.UltralyticsRunner") as runner_cls:
            runner_cls.return_value.val.return_value = result

            returncode = main([
                "val",
                "--weights",
                "runs/train/nozzle_ng_ok_v8/weights/best.pt",
                "--task",
                "test",
                "--conf",
                "0.6",
                "--dry-run",
            ])

        self.assertEqual(returncode, 0)
        runner_cls.return_value.val.assert_called_once()
        kwargs = runner_cls.return_value.val.call_args.args[0]
        self.assertEqual(kwargs["split"], "test")
        self.assertEqual(kwargs["conf"], 0.6)
        self.assertEqual(kwargs["project"], str((Path.cwd() / "runs" / "detect" / "val").resolve()))
        self.assertEqual(kwargs["name"], "nozzle_ng_ok_v8")

    def test_main_val_can_export_error_samples_after_validation(self):
        save_dir = Path("runs/detect/val/exp-conf-025")
        result = YoloResult(
            action="val",
            returncode=0,
            summary="val finished",
            payload=type("Metrics", (), {"save_dir": save_dir})(),
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.UltralyticsRunner") as runner_cls, \
             patch("project.nozzle_inspection.main.ErrorSampleExporter") as exporter_cls:
            runner_cls.return_value.val.return_value = result
            exporter_cls.return_value.export.return_value = {
                "missed_target": 1,
                "false_alarm": 2,
                "class_error": 3,
                "images_with_errors": 4,
            }

            returncode = main([
                "val",
                "--weights",
                "runs/train/nozzle_ng_ok_v8/weights/best.pt",
                "--task",
                "train",
                "--save-txt",
                "--save-conf",
                "--export-error-samples",
            ])

        self.assertEqual(returncode, 0)
        kwargs = runner_cls.return_value.val.call_args.args[0]
        self.assertEqual(kwargs["split"], "train")
        self.assertTrue(kwargs["save_txt"])
        self.assertTrue(kwargs["save_conf"])
        exporter_cls.assert_called_once()
        self.assertEqual(exporter_cls.call_args.kwargs["task"], "train")
        self.assertEqual(exporter_cls.call_args.kwargs["val_save_dir"], save_dir)
        self.assertEqual(exporter_cls.call_args.kwargs["output_root"], Path("runs/train/BadCase"))


if __name__ == "__main__":
    unittest.main()
