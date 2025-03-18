from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from lours.dataset import from_coco
from lours.utils.grouper import ContinuousGroup

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "test_data"


@pytest.mark.parametrize("inplace", [True, False])
def test_small_split(inplace):
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
        target_split_shares=[0.5, 0.5], split_names=["a", "b"], inplace=inplace
    )

    if inplace:
        assert splitted.images is test_dataset.images
    else:
        assert splitted.images is not test_dataset.images

    target_splits = pd.Series(
        [1, 1], index=pd.Index(["a", "b"], name="split"), name="count"
    )
    assert_series_equal(target_splits, splitted.images["split"].value_counts())


def test_split_bad_params():
    test_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )

    with pytest.raises(ValueError):
        test_dataset.split(split_names=["a", "b"], target_split_shares=[0, 0.5, 0.5])

    with pytest.raises(ValueError):
        test_dataset.split(split_names=["a", "b"], target_split_shares=[0.8, 0.5])

    with pytest.raises(ValueError):
        test_dataset.split(split_names=["a"], target_split_shares=[1])

    with pytest.raises(ValueError):
        test_dataset.split(split_names=[], target_split_shares=[])


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


def test_coco_partially_already_assigned_split():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json",
        images_root=Path("."),
        split="valid",
    )
    coco = coco[::100]
    gen = np.random.default_rng(1)
    coco.images["fake_category"] = gen.integers(0, 5, size=len(coco))
    coco.images["split"] = None

    # Only one element has train split assigned

    coco.images.loc[(coco.images["fake_category"] == 0).idxmax(), "split"] = "train"

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id", "fake_category"],
    )

    assert 0 not in splitted_coco.get_split("valid").images["fake_category"].to_numpy()
    assert 0 in splitted_coco.get_split("train").images["fake_category"].to_numpy()

    # All of them have split assigned
    coco.images["split"] = None
    coco.images.loc[coco.images["fake_category"] == 0, "split"] = "train"
    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=["category_id"],
    )

    assert 0 not in splitted_coco.get_split("valid").images["fake_category"].to_numpy()
    assert 0 in splitted_coco.get_split("train").images["fake_category"].to_numpy()

    # Multiple split values for same separate group, should raise a warning

    coco.images.loc[coco.images["fake_category"] == 0, "split"] = "train"
    coco.images.loc[(coco.images["fake_category"] == 0).idxmax(), "split"] = "valid"

    with pytest.warns(RuntimeWarning):
        splitted_coco = coco.split(
            input_seed=1,
            split_names=["train", "valid"],
            target_split_shares=[0.9, 0.1],
            keep_separate_groups=["image_id", "fake_category"],
            keep_balanced_groups=["category_id"],
        )


def test_coco_already_assigned_split_warning():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json",
        images_root=Path("."),
        split="valid",
    )
    coco = coco[::100]
    coco.annotations["split"] = np.random.choice(
        ["train", "valid"], coco.len_annot(), p=[0.5, 0.5]
    )
    assert coco.annotations.groupby("image_id")["split"].unique().apply(len).max() == 2

    with pytest.warns(RuntimeWarning):
        coco.split(
            input_seed=1,
            split_names=["train", "valid"],
            target_split_shares=[0.9, 0.1],
            keep_separate_groups=["image_id"],
            keep_balanced_groups=["category_id"],
        )


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
        keep_balanced_groups_weights=[1, 1],
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


@pytest.mark.parametrize("inplace", [True, False])
def test_simple_split(inplace):
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco.images.split = None

    splitted_coco = coco.split(
        input_seed=1,
        inplace=inplace,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=[],
    )

    if inplace:
        assert splitted_coco.images is coco.images
    else:
        assert splitted_coco.images is not coco.images

    split_share_target = pd.Series([0.9, 0.1], index=["train", "valid"])
    result_share = splitted_coco.images["split"].value_counts() / len(coco)

    assert_series_equal(
        split_share_target,
        result_share,
        check_exact=False,
        check_like=True,
        check_names=False,
        atol=1e-2,
    )


def test_simple_split_already_assigned():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco.images.split = None
    gen = np.random.default_rng(1)
    to_assign = gen.choice([True, False], size=len(coco), p=[0.2, 0.8])
    coco.images.loc[to_assign, "split"] = "train"

    to_assign_invalid = (gen.choice([True, False], size=len(coco), p=[0.2, 0.8])) & (
        ~to_assign
    )
    coco.images.loc[to_assign_invalid, "split"] = "invalid"

    splitted_coco = coco.split(
        input_seed=1,
        split_names=["train", "valid"],
        target_split_shares=[0.9, 0.1],
        keep_separate_groups=["image_id"],
        keep_balanced_groups=[],
    )

    split_share_target = pd.Series([0.9, 0.1], index=["train", "valid"])
    result_share = splitted_coco.images["split"].value_counts() / len(coco)

    assert_series_equal(
        split_share_target,
        result_share,
        check_exact=False,
        check_like=True,
        check_names=False,
        atol=1e-2,
    )

    assert list(coco.images.loc[to_assign, "split"].unique()) == ["train"]


def test_simple_split_too_many_already_assigned():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco.images.split = None
    gen = np.random.default_rng(1)
    to_assign = gen.choice([True, False], size=len(coco), p=[0.2, 0.8])
    coco.images.loc[to_assign, "split"] = "valid"
    with pytest.warns(RuntimeWarning):
        splitted_coco = coco.split(
            input_seed=0,
            split_names=["train", "valid", "test"],
            target_split_shares=[0.8, 0.1, 0.1],
            keep_separate_groups=["image_id"],
            keep_balanced_groups=[],
        )

    split_share_target = pd.Series(
        [0.8 * 0.8 / 0.9, 0.2, 0.1 * 0.8 / 0.9], index=["train", "valid", "test"]
    )
    result_share = splitted_coco.images["split"].value_counts() / len(coco)

    assert_series_equal(
        split_share_target,
        result_share,
        check_exact=False,
        check_like=True,
        check_names=False,
        atol=1e-2,
    )

    assert list(coco.images.loc[to_assign, "split"].unique()) == ["valid"]
    assert "valid" not in coco.images.loc[~to_assign, "split"].values
