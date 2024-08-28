import json
from pathlib import Path

import pandas as pd
import pytest

from libia.dataset import from_coco
from libia.utils.testing import (
    assert_bounding_boxes_well_formed,
    assert_columns_properly_normalized,
    assert_dataset_equal,
    assert_frame_intersections_equal,
    assert_ids_well_formed,
    assert_images_valid,
    assert_label_map_well_formed,
)

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def get_valid_dataset():
    """Get a dataset with valid data. Tests are supposed to pass with it.

    It will then be altered so that the tests fail

    Note:
        "valid" is not related to the split value, but the validity of the dataset

    Returns:
        Valid dataset from coco_dataset folder
    """
    return from_coco(
        coco_json=DATA / "coco_dataset" / "annotations_valid.json",
        images_root=DATA / "coco_dataset" / "data" / "Images",
        split="valid",
    ) + from_coco(
        coco_json=DATA / "coco_dataset" / "annotations_train.json",
        images_root=DATA / "coco_dataset" / "data" / "Images",
        split="train",
    )


def test_assert_frame_intersection_equal():
    dataset = get_valid_dataset()

    columns = dataset.annotations.columns
    index = dataset.annotations.index
    df1 = dataset.annotations.loc[index[2:], columns[:3]]  # pyright: ignore
    df2 = dataset.annotations.loc[index[:3], columns[2:]]  # pyright: ignore

    assert_frame_intersections_equal(df1, df2)

    with pytest.raises(AssertionError):
        # Setting to a different value
        df2.iloc[:, 0] = -1
        assert_frame_intersections_equal(df1, df2)


def test_assert_images_valid():
    dataset = get_valid_dataset()

    assert_images_valid(dataset)
    assert_images_valid(dataset, check_exhaustive=True)

    assert_images_valid(dataset.iloc[:-1])
    with pytest.raises(AssertionError):
        assert_images_valid(dataset.iloc[:-1], check_exhaustive=True)

    dataset.images["width"] = 0
    with pytest.raises(AssertionError):
        assert_images_valid(dataset)
    assert_images_valid(dataset, load_images=False)

    with pytest.raises(AssertionError):
        assert_images_valid(dataset, assert_is_symlink=True)

    dataset.images_root = Path(".")
    with pytest.raises(AssertionError):
        assert_images_valid(dataset)


def test_assert_bounding_boxes_well_formed():
    dataset = get_valid_dataset()

    assert_bounding_boxes_well_formed(dataset)

    first_row = dataset.annotations.index[0]
    first_row_img_width = dataset.images.loc[
        dataset.annotations.loc[first_row, "image_id"], "width"  # pyright: ignore
    ]
    first_row_img_height = dataset.images.loc[
        dataset.annotations.loc[first_row, "image_id"], "height"  # pyright: ignore
    ]

    # First test: box xmin should be >= 0
    dataset.annotations.loc[first_row, "box_x_min"] = 0
    assert_bounding_boxes_well_formed(dataset)

    dataset.annotations.loc[first_row, "box_x_min"] = -1
    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    # Second test: box ymin should be >= 0
    dataset = get_valid_dataset()

    dataset.annotations.loc[first_row, "box_y_min"] = 0
    assert_bounding_boxes_well_formed(dataset)

    dataset.annotations.loc[first_row, "box_y_min"] = -1
    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    # Third test: box width should be > 0, unless allow_keypoints is True,
    # then it should be >= 0
    dataset = get_valid_dataset()
    dataset.annotations.loc[first_row, "box_width"] = -1

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset, allow_keypoints=True)

    dataset.annotations.loc[first_row, "box_width"] = 0
    assert_bounding_boxes_well_formed(dataset, allow_keypoints=True)

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    # Fourth test: box height should be > 0, unless allow_keypoints is True,
    # the height should be >= 0
    dataset = get_valid_dataset()
    dataset.annotations.loc[first_row, "box_height"] = -1

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset, allow_keypoints=True)

    dataset.annotations.loc[first_row, "box_height"] = 0
    assert_bounding_boxes_well_formed(dataset, allow_keypoints=True)

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    # Fifth test: box xmax should be <= image width
    # xmax is actually xmin + box width
    # So we artificially set xmin to image width -1 and box width to 1 and then 2
    dataset = get_valid_dataset()
    dataset.annotations.loc[first_row, "box_x_min"] = (
        first_row_img_width - 1
    )  # pyright: ignore
    dataset.annotations.loc[first_row, "box_width"] = 1  # pyright: ignore
    assert_bounding_boxes_well_formed(dataset)

    dataset.annotations.loc[first_row, "box_width"] = 2  # pyright: ignore

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)

    # Sixth test: box ymax should be <= image height
    # ymax is actually ymin + box height
    # So we artificially set ymin to image height -1 and box width to 1 and then 2
    dataset = get_valid_dataset()
    dataset.annotations.loc[first_row, "box_y_min"] = (
        first_row_img_height - 1
    )  # pyright: ignore
    dataset.annotations.loc[first_row, "box_height"] = 1  # pyright: ignore
    assert_bounding_boxes_well_formed(dataset)

    dataset.annotations.loc[first_row, "box_height"] = 2  # pyright: ignore

    with pytest.raises(AssertionError):
        assert_bounding_boxes_well_formed(dataset)


def test_assert_dataset_equal():
    dataset = get_valid_dataset()

    assert_dataset_equal(dataset, dataset)

    dataset2 = get_valid_dataset()
    dataset2.annotations = dataset2.annotations.drop("area", axis=1)
    assert_dataset_equal(dataset, dataset2)

    dataset2 = get_valid_dataset().reset_index(10)

    with pytest.raises(AssertionError):
        assert_dataset_equal(dataset, dataset2)

    assert_dataset_equal(dataset, dataset2, ignore_index=True)

    dataset2 = get_valid_dataset()
    dataset2.annotations["area"] = 20

    with pytest.raises(AssertionError):
        assert_dataset_equal(dataset, dataset2)

    dataset2 = get_valid_dataset()
    dataset2.label_map[max(dataset2.label_map) + 1] = "new class"

    with pytest.raises(AssertionError):
        assert_dataset_equal(dataset, dataset2)

    dataset2 = get_valid_dataset()
    dataset2.label_map[list(dataset2.label_map)[0]] = "new class"

    with pytest.raises(AssertionError):
        assert_dataset_equal(dataset, dataset2)

    dataset2 = get_valid_dataset()
    dataset2.booleanized_columns["images"] = {"a", "b"}

    with pytest.raises(AssertionError):
        assert_dataset_equal(dataset, dataset2)


def test_assert_ids_well_formed():
    dataset = get_valid_dataset()

    assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.images.index.name = "other"

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.annotations.index.name = "other"

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.images.index = pd.Index(
        [
            dataset.images.index[1],
            dataset.images.index[1],
            *dataset.images.index[2:],
        ]
    )

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.annotations.index = pd.Index(
        [
            dataset.annotations.index[1],
            dataset.annotations.index[1],
            *dataset.annotations.index[2:],
        ]
    )

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.images.loc[dataset.images.index[0], "relative_path"] = dataset.images.loc[
        dataset.images.index[1], "relative_path"
    ]  # pyright: ignore

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.annotations.loc[dataset.annotations.index[0], "image_id"] = (
        max(dataset.images.index) + 1
    )

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)

    dataset = get_valid_dataset()
    dataset.annotations.loc[dataset.annotations.index[0], "category_id"] = (
        max(dataset.label_map) + 1
    )

    with pytest.raises(AssertionError):
        assert_ids_well_formed(dataset)


def test_assert_label_map_well_formed():
    dataset = get_valid_dataset()

    assert_label_map_well_formed(dataset)

    label_keys = list(dataset.label_map.keys())
    dataset.label_map[label_keys[0]] = dataset.label_map[label_keys[1]]

    with pytest.raises(AssertionError):
        assert_label_map_well_formed(dataset)


def test_assert_columns_properly_normalized():
    with open(DATA / "caipy_dataset" / "tags" / "default_schema" / "785.json") as f:
        input_dict = json.load(f)
    annotations = pd.json_normalize(input_dict["annotations"])
    assert_columns_properly_normalized(annotations)

    input_dict["annotations"][0]["attributes"] = None
    annotations = pd.json_normalize(input_dict["annotations"])
    with pytest.raises(AssertionError):
        assert_columns_properly_normalized(annotations)


def test_regression(dataset_regression):
    from libia.utils.doc_utils import dummy_dataset

    my_dataset = dummy_dataset(seed=0)
    dataset_regression.check(my_dataset)

    my_dataset2 = dummy_dataset(seed=1)
    with pytest.raises(AssertionError):
        dataset_regression.check(my_dataset2)


def test_regression_with_images(dataset_regression):
    from libia.utils.doc_utils import dummy_dataset

    my_dataset = dummy_dataset(seed=0, generate_real_images=True)
    dataset_regression.check(my_dataset, check_images=True)

    my_dataset2 = dummy_dataset(seed=1, generate_real_images=True)
    with pytest.raises(AssertionError):
        dataset_regression.check(my_dataset2, check_images=True)
