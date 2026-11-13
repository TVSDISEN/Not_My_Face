from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.data import extract_faces
from src.data.augmentations import build_train_augmentation
from src.data.extract_faces import expand_box, select_face_box
from src.data.frequency import compute_fft_spectrogram
from src.data.make_folds import assign_folds, main as make_folds_main
from src.data.manifest import build_manifest, normalize_label


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


class AugmentationTests(unittest.TestCase):
    def test_training_pipeline_keeps_image_dimensions(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        result = build_train_augmentation()(image=image)["image"]
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
