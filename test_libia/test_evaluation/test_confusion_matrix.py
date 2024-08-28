from pathlib import Path

import numpy as np
import pandas as pd

from libia.dataset import from_coco
from libia.evaluation.detection import DetectionEvaluator
from libia.evaluation.detection.util import confusion_matrix
from libia.utils.grouper import ContinuousGroup

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_group_confusion_matrix():
    data = {
        "groundtruth_label": [
            "person",
            "car",
            "person",
            "person",
            "person",
            "person",
            "car",
            "person",
            "car",
            pd.NA,
        ],
        "prediction_label": [
            "person",
            "car",
            "person",
            "person",
            "person",
            pd.NA,
            pd.NA,
            "person",
            "person",
            "person",
        ],
    }

    dataframe = pd.DataFrame(data)

    confusion_matrix_data = confusion_matrix(
        dataframe,
    )

    expected_confusion_matrix = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0, 5 / 6, 1 / 6],
            [0, 1, 0],
        ]
    )

    expected_labels = ["car", "person", "None"]

    assert np.allclose(confusion_matrix_data.values, expected_confusion_matrix)
    assert expected_labels == confusion_matrix_data.columns.tolist()


def test_confusion_matrix_identical():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    predictions = dataset
    predictions.annotations["confidence"] = 1
    evaluator = DetectionEvaluator(dataset, predictions=predictions)
    results_1 = evaluator.compute_confusion_matrix("predictions")
    results_1 = results_1.loc[results_1["model"] == "predictions"].drop(columns="model")
    results_2 = evaluator.compute_confusion_matrix("predictions", min_iou=0.8)
    results_2 = results_2.loc[results_2["model"] == "predictions"].drop(columns="model")
    expected_confusion_matrix = np.array([[1, 0], [0, 1]])

    assert np.allclose(results_1.values, expected_confusion_matrix)
    assert np.allclose(results_2.values, expected_confusion_matrix)


def test_confusion_matrix_multiple_predictions():
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
    predictions2 = from_coco(
        coco_json=DATA / "coco_dataset/predictions_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )

    evaluator = DetectionEvaluator(
        groundtruth=dataset, predictions_1=predictions, predictions_2=predictions2
    )
    results = evaluator.compute_confusion_matrix(
        ["predictions_1", "predictions_2"], min_confidence=0.6
    )

    columns = (
        results.loc[results["model"] == "predictions_2"]
        .drop(columns="model")
        .columns.tolist()
    )

    assert columns == ["hair drier", "toothbrush", "None"]
    assert sorted(results["model"].unique()) == ["predictions_1", "predictions_2"]


def test_confusion_matrix_groups():
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

    evaluator = DetectionEvaluator(groundtruth=dataset, predictions_3=predictions)
    box_height_group = ContinuousGroup(name="box_height", bins=3, qcut=True)
    results = evaluator.compute_confusion_matrix(
        "predictions_3", groups=[box_height_group]
    )
    results_grouped = (
        results.loc[results["model"] == "predictions_3"]
        .drop(columns="model")
        .groupby("box_height", observed=False)
    )
    group_names = list(results_grouped.groups.keys())

    assert len(group_names) == 3
