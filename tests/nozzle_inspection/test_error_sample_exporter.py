import tempfile
import unittest
from pathlib import Path

import yaml

from project.nozzle_inspection.evaluation.error_sample_exporter import ErrorSampleExporter


class ErrorSampleExporterTest(unittest.TestCase):
    def test_exports_class_error_sample_for_manual_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            image_dir = dataset / "images" / "train"
            label_dir = dataset / "labels" / "train"
            pred_dir = root / "runs" / "detect" / "val" / "exp" / "labels"
            output_dir = root / "runs" / "error_samples"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            image_path = image_dir / "sample.jpg"
            label_path = label_dir / "sample.txt"
            pred_path = pred_dir / "sample.txt"
            image_path.write_bytes(b"fake image")
            label_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            pred_path.write_text("1 0.5 0.5 0.4 0.4 0.91\n", encoding="utf-8")
            stale_path = output_dir / "exp" / "class_error" / "images" / "stale.json"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("stale", encoding="utf-8")

            data_yaml = root / "dataset.yaml"
            data_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(dataset),
                        "train": "images/train",
                        "names": {0: "NG", 1: "OK"},
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            summary = ErrorSampleExporter(
                data_yaml=data_yaml,
                task="train",
                val_save_dir=root / "runs" / "detect" / "val" / "exp",
                output_root=output_dir,
                iou_threshold=0.5,
            ).export()

            review_root = output_dir / "exp"
            self.assertEqual(summary["class_error"], 1)
            self.assertTrue((review_root / "class_error" / "images" / "sample.jpg").exists())
            self.assertTrue((review_root / "class_error" / "labels" / "sample.txt").exists())
            self.assertTrue((review_root / "class_error" / "predictions" / "sample.txt").exists())
            self.assertTrue((review_root / "error_summary.csv").exists())
            self.assertTrue((review_root / "badcase_config.yaml").exists())
            self.assertTrue((review_root / "badcase_summary.md").exists())
            self.assertFalse(stale_path.exists())

            summary_md = (review_root / "badcase_summary.md").read_text(encoding="utf-8")
            self.assertIn("# BadCase 复查摘要", summary_md)
            self.assertIn("类别错误", summary_md)


if __name__ == "__main__":
    unittest.main()
