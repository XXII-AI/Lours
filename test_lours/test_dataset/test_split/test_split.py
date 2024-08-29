from pathlib import Path

import pandas as pd
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from lours.dataset import from_coco
from lours.utils.grouper import ContinuousGroup

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "test_data"


def test_small_split():
    test_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    splitted = test_dataset.split()
    assert_frame_equal(test_dataset.images, splitted.images)
    assert_frame_equal(test_dataset.annotations, splitted.annotations)

    test_dataset.images["split"] = None
    splitted = test_dataset.split(
        target_split_shares=[0.5, 0.5], split_names=["a", "b"]
    )

    target_splits = pd.Series(
        [1, 1], index=pd.Index(["a", "b"], name="split"), name="count"
    )
    assert_series_equal(target_splits, splitted.images["split"].value_counts())


def test_coco_already_assigned_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json",
        images_root=Path("."),
        split="valid",
    )
    coco = coco[::10]

    box_height_group = ContinuousGroup("box_height", 10, qcut=True)

    former_split = coco.images["split"].copy()

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["category_id"],
        keep_balanced_groups=[box_height_group],
    )

    assert_series_equal(former_split, splitted_coco.images["split"])


def test_coco_balanced_category_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco = coco[::10]

    coco.images.split = None
    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=["category_id"],
    )

    result_shares_images = splitted_coco.images.groupby("split").size() / len(
        splitted_coco
    )
    assert_almost_equal([0.9, 0.1], result_shares_images[["train", "valid"]])


def test_balanced_continuous_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco = coco[::10]
    coco.images.split = None

    box_height_group = ContinuousGroup("box_height", 10, qcut=True)

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=[box_height_group],
        earth_mover_regularization=0.1,
    )
    result_shares_images = splitted_coco.images.groupby("split").size() / len(
        splitted_coco
    )
    assert_almost_equal([0.9, 0.1], result_shares_images[["train", "valid"]])


def test_balanced_mix_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    ).remap_from_preset("coco", "supercategory")
    coco = coco[::10]
    coco.images.split = None

    box_height_group = ContinuousGroup("box_height", 10, qcut=True)

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=["category_id", box_height_group],
        earth_mover_regularization=0.1,
    )
    result_shares_images = splitted_coco.images.groupby("split").size() / len(
        splitted_coco
    )
    assert_almost_equal([0.9, 0.1], result_shares_images[["train", "valid"]])


def test_balanced_two_continuous_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco = coco[::10]
    coco.images.split = None

    box_height_group = ContinuousGroup("box_height", 10, qcut=True)
    box_width_group = ContinuousGroup("box_width", 10, qcut=True)

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=[box_width_group, box_height_group],
        earth_mover_regularization=0,
    )
    result_shares_images = splitted_coco.images.groupby("split").size() / len(
        splitted_coco
    )
    assert_almost_equal([0.9, 0.1], result_shares_images[["train", "valid"]])
