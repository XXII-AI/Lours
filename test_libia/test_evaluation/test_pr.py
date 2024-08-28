from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from libia.dataset import from_coco
from libia.evaluation.detection import DetectionEvaluator

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_cocoapi_matches():
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
    pr, aps = evaluator.compute_precision_recall(
        ious=[0, 0.2, 0.5], groups="category_id"
    )
    coco_gt = COCO(DATA / "coco_dataset/annotations_valid_random.json")
    coco_dt = COCO(DATA / "coco_dataset/predictions_valid_random.json")
    coco_evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    coco_evaluator.params.iouThrs = np.array([1e-5, 0.2, 0.5])
    # pyright ignore Will soon be unnecessary
    # See https://github.com/python/typeshed/pull/9897
    coco_evaluator.params.areaRng = [[0, 1e5**2]]  # pyright: ignore
    coco_evaluator.params.maxDets = [100]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    evaluation = coco_evaluator.eval["precision"][:, :, :, 0, 0]  # pyright: ignore
    print(evaluation)
    print(aps)


def test_pr_different_label_maps():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    predictions = from_coco(
        coco_json=DATA / "coco_dataset/predictions_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    ).remap_classes({0: 2})
    # dataset and predictions have different label maps
    # This should only raise a warning, and assume that there are only false negative
    # for class id 0 (no prediction),
    # and only false positive for class id 2 (no groundtruth)
    evaluator = DetectionEvaluator(dataset, predictions=predictions)
    pr, aps = evaluator.compute_precision_recall(
        ious=[0, 0.2, 0.5], groups="category_id"
    )
    # PR curve should have 3 categories, with 2 of them set to 0 almost everywhere
    assert set(pr["category_id"].unique()) == {0, 1, 2}
