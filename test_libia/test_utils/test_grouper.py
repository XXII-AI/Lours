from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pytest import raises

from libia.dataset import from_coco
from libia.utils import grouper

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_make_root_data_pandas_compatible():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    data = coco.annotations
    root_data = coco.images
    np.random.seed(0)
    root_data["fake_id"] = np.random.randint(0, 10, len(root_data))
    root_data["fake_size"] = 100 * np.random.random(len(root_data))
    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data, root_data=root_data, key_to_root="image_id", g="fake_id"
    )
    assert group_name == "fake_id"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert is_category

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data,
        root_data=root_data,
        key_to_root="image_id",
        g=grouper.ContinuousGroup("fake_size", 10),
    )

    assert group_name == "fake_size"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert len(pandas_group.unique()) == 10
    assert not is_category

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data,
        root_data=root_data,
        key_to_root="image_id",
        g=grouper.ContinuousGroup("fake_size", 10, log=True, qcut=False),
    )

    assert group_name == "fake_size"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert len(pandas_group.unique()) == 10
    assert not is_category


def test_make_data_pandas_compatible():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    data = coco.annotations
    data["fake_id"] = np.random.randint(0, 10, len(data))
    data["fake_size"] = np.random.randn(len(data))
    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data, g="fake_id"
    )
    assert group_name == "fake_id"
    assert pandas_group == "fake_id"
    assert is_category

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10)
    )

    assert group_name == "fake_size"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert len(pandas_group.unique()) == 10
    assert not is_category

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data,
        g=grouper.ContinuousGroup("fake_size", bins=[-float("inf"), 0, float("inf")]),
    )

    assert group_name == "fake_size"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert len(pandas_group.unique()) == 2
    assert not is_category

    # This should fail because "fake_size" values are sometimes negative
    with raises(ValueError):
        grouper.make_pandas_compatible(
            data=data,
            g=grouper.ContinuousGroup("fake_size", 10, log=True),
        )

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data,
        g=grouper.ContinuousGroup("fake_size", 10, log=False, qcut=True),
    )

    assert group_name == "fake_size"
    assert isinstance(pandas_group, pd.Series)
    assert len(pandas_group) == len(data)
    assert len(pandas_group.unique()) == 10
    assert not is_category


def test_label_types():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    data = coco.annotations
    data["fake_id"] = np.random.randint(0, 10, len(data))
    data["fake_size"] = np.random.randn(len(data))

    pandas_group = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="mid")
    )[1]
    target = np.array(
        [
            -3.7345,
            -2.941,
            -2.1515,
            -1.362,
            -0.5725,
            0.217,
            1.0065,
            1.796,
            2.5855,
            3.375,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)

    pandas_group = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="mean")
    )[1]

    target = np.array(
        [
            -3.57622866,
            -2.81813268,
            -2.03869917,
            -1.29311724,
            -0.54028369,
            0.20698126,
            0.95833345,
            1.71192913,
            2.45991059,
            3.2138484,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="median")
    )

    target = np.array(
        [
            -3.50361545,
            -2.76894588,
            -1.99502576,
            -1.264642,
            -0.52548114,
            0.20053149,
            0.9358896,
            1.67847603,
            2.40298958,
            3.16176182,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)
