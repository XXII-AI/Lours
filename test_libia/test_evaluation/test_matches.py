from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from libia.dataset import from_coco
from libia.evaluation.detection import DetectionEvaluator

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_matches_identical():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    predictions = deepcopy(dataset)
    predictions.annotations["confidence"] = 1
    evaluator = DetectionEvaluator(dataset, predictions=predictions)
    matches = evaluator.compute_matches()["predictions"]
    unique_ious = matches.iou.unique()
    assert np.allclose(unique_ious, 1)


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
    matches = evaluator.compute_matches(min_iou=0)["predictions"]
    libia_gt_matches = (
        matches.dropna(subset=["groundtruth_id"])
        .set_index("groundtruth_id")["prediction_id"]
        .fillna(0)
        .sort_index()
    )
    libia_dt_matches = (
        matches.dropna(subset=["prediction_id"])
        .set_index("prediction_id")["groundtruth_id"]
        .fillna(0)
        .sort_index()
    )
    libia_gt_matches = libia_gt_matches.astype(int)
    libia_gt_matches.index = libia_gt_matches.index.astype(int)
    libia_dt_matches = libia_dt_matches.astype(int)
    libia_dt_matches.index = libia_dt_matches.index.astype(int)
    coco_gt = COCO(DATA / "coco_dataset/annotations_valid_random.json")
    coco_dt = COCO(DATA / "coco_dataset/predictions_valid_random.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([1e-5])
    # pyright ignore Will soon be unnecessary
    # See https://github.com/python/typeshed/pull/9897
    coco_eval.params.areaRng = [[0, 1e5**2]]  # pyright: ignore
    coco_eval.evaluate()
    cocodf = pd.DataFrame([*filter(lambda x: x is not None, coco_eval.evalImgs)])
    coco_gt_matches = np.concatenate(
        [gt_id[0] for gt_id in cocodf["gtMatches"].values]
    ).astype(int)
    coco_dt_matches = np.concatenate(
        [dt_id[0] for dt_id in cocodf["dtMatches"].values]
    ).astype(int)
    coco_gt_ids = np.concatenate([*cocodf["gtIds"].values]).astype(int)
    coco_dt_ids = np.concatenate([*cocodf["dtIds"].values]).astype(int)
    coco_gt_matches = pd.Series(coco_gt_matches, index=coco_gt_ids).sort_index()
    coco_dt_matches = pd.Series(coco_dt_matches, index=coco_dt_ids).sort_index()
    assert all(libia_gt_matches == coco_gt_matches)
    assert all(libia_dt_matches == coco_dt_matches)
