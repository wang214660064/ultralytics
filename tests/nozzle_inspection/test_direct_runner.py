import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig


class DirectRunnerTest(unittest.TestCase):
    def test_train_action_calls_ultralytics_directly(self):
        model = Mock()
        yolo_class = Mock(return_value=model)
        ultralytics_module = SimpleNamespace(YOLO=yolo_class)
        config = RunConfig(action="train", epochs=1, workers=0, train_name="direct-train")

        with patch.dict(sys.modules, {"ultralytics": ultralytics_module}), \
             patch.object(run_project, "main", return_value=0, create=True), \
             patch.object(run_project, "_detect_device", return_value="cpu"):
            returncode = run_project.run(config)

        self.assertEqual(returncode, 0)
        model.train.assert_called_once()
        self.assertEqual(model.train.call_args.kwargs["epochs"], 1)

    def test_val_action_calls_ultralytics_directly(self):
        metrics = SimpleNamespace(save_dir="runs/val/direct-val")
        model = Mock()
        model.val.return_value = metrics
        yolo_class = Mock(return_value=model)
        ultralytics_module = SimpleNamespace(YOLO=yolo_class)
        config = RunConfig(
            action="val",
            eval_task="val",
            val_name="direct-val",
            export_error_samples=False,
        )

        with patch.dict(sys.modules, {"ultralytics": ultralytics_module}), \
             patch.object(run_project, "main", return_value=0, create=True), \
             patch.object(run_project, "_detect_device", return_value="cpu"):
            returncode = run_project.run(config)

        self.assertEqual(returncode, 0)
        model.val.assert_called_once()
        self.assertEqual(model.val.call_args.kwargs["split"], "val")


if __name__ == "__main__":
    unittest.main()
