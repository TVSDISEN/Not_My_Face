from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from src.training import run_training
from src.training.run_training import _validate_training_manifest


class TrainingManifestTests(unittest.TestCase):
    def _write_manifest(self, directory, rows):
        manifest_path = Path(directory) / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return manifest_path

    def test_manifest_requires_all_folds_and_binary_labels(self):
        rows = [
            {
                "rgb_path": f"rgb-{fold}-{label}.jpg",
                "freq_path": f"freq-{fold}-{label}.png",
                "fold": fold,
                "label": label,
            }
            for fold in range(5)
            for label in (0, 1)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(temporary_directory, rows)

            result = _validate_training_manifest(manifest_path, num_folds=5)

        self.assertEqual(set(result["fold"]), {0, 1, 2, 3, 4})
        self.assertEqual(set(result["label"]), {0, 1})

    def test_manifest_rejects_missing_fold(self):
        rows = [
            {
                "rgb_path": f"rgb-{fold}.jpg",
                "freq_path": f"freq-{fold}.png",
                "fold": fold,
                "label": fold % 2,
            }
            for fold in range(4)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(temporary_directory, rows)

            with self.assertRaisesRegex(ValueError, "every fold"):
                _validate_training_manifest(manifest_path, num_folds=5)

    def test_manifest_rejects_non_binary_labels(self):
        rows = [
            {
                "rgb_path": f"rgb-{fold}.jpg",
                "freq_path": f"freq-{fold}.png",
                "fold": fold,
                "label": 2 if fold == 0 else fold % 2,
            }
            for fold in range(5)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(temporary_directory, rows)

            with self.assertRaisesRegex(ValueError, "binary"):
                _validate_training_manifest(manifest_path, num_folds=5)

    def test_runner_saves_lowest_validation_loss_checkpoint_for_each_fold(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        rows = [
            {
                "rgb_path": f"rgb-{fold}-{label}.jpg",
                "freq_path": f"freq-{fold}-{label}.png",
                "fold": fold,
                "label": label,
            }
            for fold in range(5)
            for label in (0, 1)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self._write_manifest(root, rows)
            checkpoint_dir = root / "checkpoints"
            losses = [value for _ in range(5) for value in (0.8, 0.4)]

            def train_once(model, dataloader, criterion, optimizer, scaler, device):
                optimizer.step()
                return 0.5, 0.5

            with patch.object(
                run_training, "DualStreamNet", side_effect=lambda **_: TinyModel()
            ), patch.object(
                run_training, "train_one_epoch", side_effect=train_once
            ), patch.object(
                run_training,
                "validate",
                side_effect=[(loss, 0.6) for loss in losses],
            ), redirect_stdout(StringIO()):
                run_training.run_kfold_training(
                    manifest_csv=manifest_path,
                    epochs=2,
                    batch_size=2,
                    num_workers=0,
                    checkpoint_dir=checkpoint_dir,
                    device="cpu",
                    pretrained=False,
                )

            checkpoints = sorted(checkpoint_dir.glob("best_model_fold_*.pt"))
            history = pd.read_csv(checkpoint_dir / "training_history_fold_0.csv")
            checkpoint = torch.load(checkpoints[0], map_location="cpu")

        self.assertEqual(len(checkpoints), 5)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["val_loss"], 0.4)
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
