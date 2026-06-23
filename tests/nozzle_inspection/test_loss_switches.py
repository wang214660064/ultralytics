import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou


class LossSwitchesTest(unittest.TestCase):
    def test_siou_is_finite_and_supports_backward(self):
        prediction = torch.tensor([[0.3, 0.3, 0.7, 0.6]], requires_grad=True)
        target = torch.tensor([[0.4, 0.35, 0.75, 0.65]])

        siou = bbox_iou(prediction, target, xywh=False, SIoU=True).mean()
        (1.0 - siou).backward()

        self.assertTrue(torch.isfinite(siou))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_ultralytics_config_accepts_siou(self):
        cfg = get_cfg(overrides={"iou_type": "siou"})

        self.assertEqual(cfg.iou_type, "siou")

    def test_bbox_loss_keeps_selected_iou_type(self):
        loss = BboxLoss(reg_max=16, iou_type="siou")

        self.assertEqual(loss.iou_type, "siou")

    def test_run_config_passes_iou_type_to_training(self):
        model = Mock()
        ultralytics_module = SimpleNamespace(YOLO=Mock(return_value=model))
        config = RunConfig(action="train", epochs=1, workers=0, iou_type="siou")

        with patch.dict(sys.modules, {"ultralytics": ultralytics_module}), \
             patch.object(run_project, "_detect_device", return_value="cpu"):
            returncode = run_project.run(config)

        self.assertEqual(returncode, 0)
        self.assertEqual(model.train.call_args.kwargs["iou_type"], "siou")


if __name__ == "__main__":
    unittest.main()
