from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from pytest import raises

from libia.dataset import from_coco
from libia.dataset.split import balanced_groups

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "test_data"


def test_hist1():
    test_df = pd.DataFrame(np.random.randint(0, 3, (10, 3)))
    test_df["split"] = 0
    test_df.loc[5:, "split"] = 1

    hist = balanced_groups.df_to_hist(test_df, "split")
    hist = hist / hist.sum()
    target_hist = pd.Series(
        [0.5, 0.5], index=pd.Index([0, 1], name="split"), name="histogram"
    )

    assert_series_equal(hist, target_hist)


def test_hist2():
    test_df = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
    ).annotations

    box_height = pd.cut(test_df["box_height"], 2)
    hist = balanced_groups.df_to_hist(test_df, ["category_id", box_height])
    hist = hist / hist.sum()

    target_hist = pd.read_csv(DATA / "coco_dataset/histogram_annotations_valid.csv")

    def interval_from_str(input_string: str) -> pd.Interval:
        numbers = input_string[1:-1].split(", ")
        return pd.Interval(float(numbers[0]), float(numbers[1]))

    height_categories = (
        target_hist["box_height"]
        .apply(interval_from_str)  # pyright: ignore
        .astype("category")
    )
    target_hist["box_height"] = height_categories.cat.as_ordered()
    target_hist = target_hist.set_index(["category_id", "box_height"])["histogram"]

    assert_series_equal(hist, target_hist)


def test_dataset_share1():
    test_share1 = pd.Series([0, 1, 0])
    test_share2 = pd.Series([1, 0, 0])
    same_share_cost = balanced_groups.dataset_share_distance(test_share1, test_share1)
    assert same_share_cost == 0
    share_cost = balanced_groups.dataset_share_distance(test_share1, test_share2)
    assert share_cost == 1
    return


def test_dataset_share2():
    test_share1 = pd.Series(np.arange(10))
    test_share1 /= test_share1.sum()
    test_share2 = pd.Series(np.arange(10)[::-1])
    test_share2 /= test_share2.sum()
    share_cost = balanced_groups.dataset_share_distance(test_share1, test_share2)
    print(share_cost)
    assert share_cost == 5 / 7
    return


def test_check_group():
    test_df = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
    ).annotations

    box_height = pd.cut(test_df["box_height"], 2).apply(lambda x: x.mid).astype(float)
    hist = balanced_groups.df_to_hist(test_df, ["category_id", box_height])
    balanced_groups.check_groups(
        hist,
        pd.Series([1], index=["category_id"]),
        pd.Series([1], index=["box_height"]),
    )

    with raises(AssertionError):
        balanced_groups.check_groups(
            hist, pd.Series([1], index=["category_id"]), pd.Series([])
        )


def test_earth_mover_distance():
    test_df = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
    ).annotations
    box_height = pd.cut(test_df["box_height"], 2).apply(lambda x: x.mid).astype(float)
    hist = balanced_groups.df_to_hist(test_df, box_height)
    emd = balanced_groups.earth_mover_distance(
        hist,
        hist,
        continuous_weights=pd.Series([1], index=["box_height"]),
        sinkhorn_lambda=1e-2,
    )
    assert np.allclose(emd, 0)

    modified_hist = hist * np.array([0.5, 2])
    emd = balanced_groups.earth_mover_distance(
        hist,
        modified_hist,
        continuous_weights=pd.Series([1], index=["box_height"]),
        sinkhorn_lambda=1e-2,
    )
    assert np.allclose(emd, 0.24762)


def test_hist_distance():
    test_df = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid_random.json",
        images_root=DATA / "coco_dataset/data",
    ).annotations

    box_height = pd.cut(test_df["box_height"], 2).apply(lambda x: x.mid).astype(float)
    hist = balanced_groups.df_to_hist(test_df, ["category_id", box_height])
    category_weights = pd.Series([1], index=["category_id"])
    continuous_weights = pd.Series([1], index=["box_height"])
    distance_with_self = balanced_groups.hist_distance(
        hist,
        hist,
        category_weights,
        continuous_weights,
        sinkhorn_lambda=1e-2,
    )
    assert np.allclose(distance_with_self, 0)

    test_df2 = from_coco(
        coco_json=DATA / "coco_dataset/predictions_valid_random.json",
        images_root=DATA / "coco_dataset/data",
    ).annotations

    box_height = pd.cut(
        pd.concat([test_df2["box_height"], test_df["box_height"]]), 2
    ).apply(lambda x: x.mid)

    hist1 = balanced_groups.df_to_hist(
        test_df, ["category_id", box_height.loc[test_df.index]]
    )
    hist2 = balanced_groups.df_to_hist(
        test_df2, ["category_id", box_height.loc[test_df2.index]]
    )

    distance_annotations_predictions = balanced_groups.hist_distance(
        hist1, hist2, category_weights, continuous_weights
    )
    assert np.allclose(distance_annotations_predictions, 0.226855)
