from pathlib import Path

import numpy as np

from libia.dataset import from_coco, from_coco_keypoints
from libia.evaluation.detection import CrowdDetectionEvaluator

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_simple_crowd_mae():
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
    evaluator = CrowdDetectionEvaluator(dataset, predictions=predictions)
    counts, _ = evaluator.compute_count_error()
    assert np.isclose(counts["absolute", "MAE"].min(), 0.5)
    assert np.isclose(counts["absolute", "RMSE"].min(), 0.7071)


def test_coco_crowd_mae():
    gt_dataset = from_coco_keypoints(coco_json=DATA / "coco_eval/instances_crowd.json")
    predictions = from_coco_keypoints(
        coco_json=DATA / "coco_eval/instances_crowd_predictions.json"
    )
    evaluator = CrowdDetectionEvaluator(gt_dataset, predictions=predictions)
    mae, _ = evaluator.compute_count_error(groups=())
    best_threshold = mae["absolute", "MAE"].idxmin()
    assert best_threshold == 0.55
    assert np.isclose(
        mae.loc[best_threshold, ("absolute", "MAE")],  # pyright: ignore
        69.097,
    )
