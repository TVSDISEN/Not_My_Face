from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

from src.training import inference_pipeline
from src.data import extract_faces
from src.data.augmentations import build_train_augmentation
from src.data.extract_faces import expand_box, select_face_box
from src.data.frequency import compute_fft_spectrogram, frequency_map_to_rgb
from src.data.make_folds import assign_folds, main as make_folds_main
from src.data.manifest import build_manifest, normalize_label
from src.data.prepare_manifests import prepare_manifests


class ManifestTests(unittest.TestCase):
    def test_auto_manifest_maps_nested_class_and_video_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for class_name in ("real", "fake"):
                for video in ("video_1", "video_2"):
                    image = root / class_name / video / "frame_0001.jpg"
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b"synthetic test image")

            manifest = build_manifest(root)

        self.assertEqual(len(manifest), 4)
        self.assertEqual(set(manifest["label"]), {0, 1})
        self.assertEqual(set(manifest["group_id"]), {"video_1", "video_2"})
        self.assertTrue(all(not Path(path).is_absolute() for path in manifest["path"]))

    def test_metadata_rejects_paths_outside_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "raw"
            root.mkdir()
            outside = Path(temporary_directory) / "outside.jpg"
            outside.write_bytes(b"synthetic test image")
            metadata = Path(temporary_directory) / "manifest.csv"
            pd.DataFrame(
                [{"path": str(outside), "label": "real", "group_id": "subject"}]
            ).to_csv(metadata, index=False)

            with self.assertRaisesRegex(ValueError, "inside the data directory"):
                build_manifest(root, metadata)

    def test_label_normalization_rejects_unknown_labels(self):
        self.assertEqual(normalize_label("Real"), 0)
        self.assertEqual(normalize_label("1"), 1)
        self.assertEqual(normalize_label(0.0), 0)
        self.assertEqual(normalize_label("1.0"), 1)
        with self.assertRaisesRegex(ValueError, "Unsupported label"):
            normalize_label("unknown")


class FoldTests(unittest.TestCase):
    def test_groups_never_cross_folds_and_each_row_is_assigned(self):
        rows = []
        for label in (0, 1):
            for group in range(10):
                rows.append(
                    {
                        "path": f"{label}/{group}/frame.jpg",
                        "label": label,
                        "group_id": f"video-{group}",
                    }
                )
        result = assign_folds(pd.DataFrame(rows), n_splits=5, seed=7)

        self.assertEqual(set(result["fold"]), {0, 1, 2, 3, 4})
        self.assertTrue((result.groupby("group_id")["fold"].nunique() == 1).all())
        self.assertTrue((result.groupby("fold")["label"].nunique() == 2).all())

    def test_insufficient_groups_fails_with_a_clear_error(self):
        manifest = pd.DataFrame(
            [
                {"path": "a", "label": 0, "group_id": "real-1"},
                {"path": "b", "label": 1, "group_id": "fake-1"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least 5 distinct groups"):
            assign_folds(manifest, n_splits=5, seed=0)

    def test_cli_writes_a_fold_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "raw"
            for label in ("real", "fake"):
                for group in range(5):
                    image = data_dir / label / f"{label}_{group}" / "frame.jpg"
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b"synthetic test image")
            output_path = root / "folds.csv"
            with patch(
                "sys.argv",
                [
                    "make_folds",
                    "--data",
                    str(data_dir),
                    "--out",
                    str(output_path),
                ],
            ), redirect_stdout(StringIO()):
                make_folds_main()

            result = pd.read_csv(output_path)

        self.assertEqual(len(result), 10)
        self.assertEqual(set(result["fold"]), {0, 1, 2, 3, 4})
        self.assertTrue((result.groupby("group_id")["fold"].nunique() == 1).all())


class FaceGeometryTests(unittest.TestCase):
    def test_largest_confident_face_is_selected(self):
        boxes = np.array([[0, 0, 10, 10], [2, 2, 22, 22]], dtype=float)
        confidence = np.array([0.99, 0.95])
        self.assertEqual(
            select_face_box(boxes, confidence, min_confidence=0.9),
            (2.0, 2.0, 22.0, 22.0),
        )
        self.assertIsNone(select_face_box(boxes, confidence, min_confidence=1.0))

    def test_margin_is_expanded_and_clamped(self):
        self.assertEqual(
            expand_box((5, 5, 15, 15), (20, 20, 3), margin=0.2),
            (3, 3, 17, 17),
        )
        self.assertEqual(
            expand_box((0, 0, 10, 10), (12, 12, 3), margin=0.5),
            (0, 0, 12, 12),
        )

    def test_extraction_cli_writes_crop_and_preserves_metadata(self):
        class FakeDetector:
            def detect(self, image):
                return np.array([[10, 10, 30, 30]]), np.array([0.99])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "raw"
            source = data_dir / "frames" / "frame_1.jpg"
            source.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(source), np.full((40, 40, 3), 127, np.uint8)))
            metadata = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "path": "frames/frame_1.jpg",
                        "label": "fake",
                        "group_id": "subject_1",
                        "fold": 3,
                    }
                ]
            ).to_csv(metadata, index=False)
            output_dir = root / "faces"
            arguments = [
                "extract_faces",
                "--data",
                str(data_dir),
                "--out",
                str(output_dir),
                "--metadata",
                str(metadata),
            ]
            with patch("sys.argv", arguments), patch.object(
                extract_faces, "create_mtcnn", return_value=FakeDetector()
            ), redirect_stdout(StringIO()):
                extract_faces.main()

            manifest = pd.read_csv(output_dir / "manifest.csv")
            crop = cv2.imread(str(output_dir / "frames" / "frame_1.jpg"))

        self.assertEqual(manifest.loc[0, "label"], 1)
        self.assertEqual(manifest.loc[0, "group_id"], "subject_1")
        self.assertEqual(manifest.loc[0, "fold"], 3)
        self.assertEqual(manifest.loc[0, "original_path"], "frames/frame_1.jpg")
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (26, 26))

    def test_unlabeled_extraction_preserves_test_filename(self):
        class FakeDetector:
            def detect(self, image):
                return np.array([[5, 5, 25, 25]]), np.array([0.99])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "test_raw"
            source = data_dir / "batch_1" / "frame_1.jpg"
            source.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(source), np.full((32, 32, 3), 127, np.uint8)))
            output_dir = root / "test_faces"
            with patch(
                "sys.argv",
                [
                    "extract_faces",
                    "--data",
                    str(data_dir),
                    "--out",
                    str(output_dir),
                    "--unlabeled",
                ],
            ), patch.object(
                extract_faces, "create_mtcnn", return_value=FakeDetector()
            ), redirect_stdout(StringIO()):
                extract_faces.main()

            manifest = pd.read_csv(output_dir / "manifest.csv")

        self.assertEqual(manifest.loc[0, "filename"], "batch_1/frame_1.jpg")
        self.assertNotIn("label", manifest.columns)
        self.assertEqual(manifest.loc[0, "group_id"], "batch_1")


class ModelManifestIntegrationTests(unittest.TestCase):
    def test_model_manifests_and_frequency_cache_are_ready_to_load(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_root = root / "train_faces"
            test_root = root / "test_faces"
            train_rows = []
            for label in (0, 1):
                for group in range(5):
                    relative = Path(f"class_{label}") / f"video_{group}" / "frame.jpg"
                    image_path = train_root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(
                        cv2.imwrite(
                            str(image_path),
                            np.full((16, 20, 3), 30 + label * 180 + group, np.uint8),
                        )
                    )
                    train_rows.append(
                        {
                            "path": relative.as_posix(),
                            "label": label,
                            "group_id": f"video_{group}",
                            "fold": group % 5,
                            "original_path": f"raw/{relative.as_posix()}",
                        }
                    )
            train_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(train_rows).to_csv(train_root / "manifest.csv", index=False)

            test_rows = []
            for name in ("clip_a/frame.jpg", "clip_b/frame.jpg"):
                image_path = test_root / name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(
                    cv2.imwrite(str(image_path), np.full((16, 20, 3), 120, np.uint8))
                )
                test_rows.append(
                    {
                        "path": name,
                        "label": 0,
                        "group_id": Path(name).parent.as_posix(),
                        "filename": name,
                        "original_path": name,
                    }
                )
            pd.DataFrame(test_rows).to_csv(test_root / "manifest.csv", index=False)

            manifest_dir = root / "manifests"
            frequency_dir = root / "frequency"
            train_output, test_output = prepare_manifests(
                train_root, test_root, manifest_dir, frequency_dir
            )
            train_manifest = pd.read_csv(train_output)
            test_manifest = pd.read_csv(test_output)

            self.assertEqual(
                list(train_manifest.columns),
                ["rgb_path", "freq_path", "fold", "label"],
            )
            self.assertEqual(
                list(test_manifest.columns), ["rgb_path", "freq_path", "filename"]
            )
            self.assertEqual(len(train_manifest), 10)
            self.assertEqual(len(test_manifest), 2)
            self.assertTrue(
                all(Path(path).is_absolute() for path in train_manifest["rgb_path"])
            )
            self.assertTrue(
                all(Path(path).is_file() for path in train_manifest["rgb_path"])
            )
            self.assertTrue(
                all(Path(path).is_file() for path in train_manifest["freq_path"])
            )
            self.assertTrue(
                all(Path(path).is_file() for path in test_manifest["rgb_path"])
            )
            self.assertTrue(
                all(Path(path).is_file() for path in test_manifest["freq_path"])
            )
            self.assertEqual(
                set(test_manifest["filename"]), {"clip_a/frame.jpg", "clip_b/frame.jpg"}
            )
            frequency = cv2.imread(test_manifest.loc[0, "freq_path"])
            self.assertEqual(frequency.shape, (16, 20, 3))

            from src.training.run_training import HackathonDataset

            test_item = HackathonDataset(test_manifest)[0]
            self.assertEqual(tuple(test_item[0].shape), (3, 256, 256))
            self.assertEqual(tuple(test_item[1].shape), (3, 256, 256))
            self.assertEqual(float(test_item[2]), 0.0)
            with patch(
                "src.training.run_training.compute_fft_spectrogram",
                wraps=compute_fft_spectrogram,
            ) as fft_transform:
                augmented_item = HackathonDataset(train_manifest, augment=True)[0]
            self.assertEqual(tuple(augmented_item[0].shape), (3, 256, 256))
            self.assertEqual(tuple(augmented_item[1].shape), (3, 256, 256))
            self.assertEqual(fft_transform.call_count, 1)

    def test_test_manifest_runs_through_batched_ensemble_inference(self):
        class FakeModel(torch.nn.Module):
            def forward(self, rgb, freq):
                return torch.zeros((rgb.shape[0], 1), device=rgb.device)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save({"model_state_dict": {}}, checkpoint_dir / "fold_0.pth")

            rows = []
            for index in range(3):
                rgb_path = root / f"rgb_{index}.jpg"
                freq_path = root / f"freq_{index}.png"
                self.assertTrue(
                    cv2.imwrite(str(rgb_path), np.full((16, 16, 3), 100, np.uint8))
                )
                self.assertTrue(
                    cv2.imwrite(str(freq_path), np.full((16, 16, 3), 100, np.uint8))
                )
                rows.append(
                    {
                        "rgb_path": str(rgb_path),
                        "freq_path": str(freq_path),
                        "filename": f"video/frame_{index}.jpg",
                    }
                )
            test_manifest = root / "test_manifest.csv"
            pd.DataFrame(rows).to_csv(test_manifest, index=False)
            output_path = root / "submission.csv"

            with patch.object(
                inference_pipeline, "DualStreamNet", return_value=FakeModel()
            ), redirect_stdout(StringIO()):
                inference_pipeline.run_ensemble_inference(
                    test_manifest_csv=str(test_manifest),
                    checkpoint_dir=str(checkpoint_dir),
                    output_csv=str(output_path),
                    batch_size=2,
                )

            submission = pd.read_csv(output_path)

        self.assertEqual(
            submission["filename"].tolist(),
            [f"video/frame_{index}.jpg" for index in range(3)],
        )
        self.assertTrue(np.allclose(submission["label"], 0.5))


class FrequencyTests(unittest.TestCase):
    def test_fft_output_is_finite_normalized_and_preserves_shape(self):
        image = np.zeros((32, 24, 3), dtype=np.uint8)
        image[8:24, 6:18] = (220, 100, 30)

        result = compute_fft_spectrogram(image)

        self.assertEqual(result.shape, image.shape[:2])
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result.max()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_fft_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "Expected a grayscale"):
            compute_fft_spectrogram(np.zeros((5, 5, 2)))
        with self.assertRaisesRegex(ValueError, "empty image"):
            compute_fft_spectrogram(np.zeros((0, 5)))

    def test_frequency_map_is_quantized_to_three_channels(self):
        source = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        result = frequency_map_to_rgb(source)

        self.assertEqual(result.shape, (2, 2, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result[..., 0], result[..., 1]))
        self.assertTrue(np.array_equal(result[..., 1], result[..., 2]))
        self.assertTrue(np.array_equal(result[0, :, 0], [0, 128]))


class AugmentationTests(unittest.TestCase):
    def test_training_pipeline_keeps_image_dimensions(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        result = build_train_augmentation()(image=image)["image"]
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
