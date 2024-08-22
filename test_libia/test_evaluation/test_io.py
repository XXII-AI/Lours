import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import fiftyone as fo
import pytest
from pandas.testing import assert_frame_equal

from libia.dataset import from_coco
from libia.evaluation import Evaluator
from libia.evaluation.detection import CrowdDetectionEvaluator, DetectionEvaluator

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_fiftyone():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    predictions = from_coco(
        coco_json=DATA / "coco_dataset/predictions_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    evaluator = DetectionEvaluator(dataset, predictions=predictions)
    fo_dataset = evaluator.to_fiftyone("eval")

    assert isinstance(fo_dataset, fo.Dataset)
    assert fo_dataset.name == "eval"
    assert len(fo_dataset) == len(evaluator.images)  # pyright: ignore
    gt_detections = fo_dataset.count_values("groundtruth_detection.detections.label")
    pred_detections = fo_dataset.count_values("predictions_detection.detections.label")
    assert isinstance(gt_detections, dict)
    assert isinstance(pred_detections, dict)
    assert len(evaluator.groundtruth) == sum(gt_detections.values())
    assert len(evaluator.predictions_dictionary["predictions"]) == sum(
        pred_detections.values()
    )


def test_parquet():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    predictions = from_coco(
        coco_json=DATA / "coco_dataset/predictions_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    evaluator = DetectionEvaluator(dataset, predictions=predictions)
    evaluator.compute_matches(category_agnostic=True)
    evaluator.compute_matches(category_agnostic=False)

    with TemporaryDirectory() as t:
        evaluator.to_parquet(t)
        evaluator2 = DetectionEvaluator.from_parquet(t)
        # The saved object is a Detection Evaluator and noting else. Should not work
        # Evaluator (parent class) nor CrowdDetectionEvaluator (child class)
        with pytest.raises(ValueError):
            Evaluator.from_parquet(t)
        with pytest.raises(ValueError):
            CrowdDetectionEvaluator.from_parquet(t)

    assert_frame_equal(evaluator.groundtruth, evaluator2.groundtruth)

    assert set(evaluator.predictions_dictionary) == set(
        evaluator2.predictions_dictionary
    )
    assert_frame_equal(
        evaluator.predictions_dictionary["predictions"],
        evaluator2.predictions_dictionary["predictions"],
    )

    assert set(evaluator.matches) == set(evaluator2.matches)

    assert_frame_equal(
        evaluator.matches["category_specific"]["predictions"],
        evaluator2.matches["category_specific"]["predictions"],
    )

    assert_frame_equal(
        evaluator.matches["category_agnostic"]["predictions"],
        evaluator2.matches["category_agnostic"]["predictions"],
    )
