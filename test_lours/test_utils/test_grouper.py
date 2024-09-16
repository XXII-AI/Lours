from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from pytest import raises

from lours.dataset import from_coco
from lours.utils import grouper

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
    np.random.seed(0)
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
    np.random.seed(0)
    data["fake_id"] = np.random.randint(0, 10, len(data))
    data["fake_size"] = np.random.randn(len(data))

    pandas_group = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="mid")
    )[1]
    target = np.array(
        [
            -3.542,
            -2.74,
            -1.9415,
            -1.143,
            -0.3448,
            0.4537,
            1.252,
            2.0505,
            2.849,
            3.647,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)

    pandas_group = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="mean")
    )[1]

    target = np.array(
        [
            -3.3643091,
            -2.5960633,
            -1.8442534,
            -1.0818792,
            -0.3264371,
            0.4304018,
            1.1904581,
            1.9497437,
            2.7433336,
            3.4620295,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)

    group_name, pandas_group, is_category = grouper.make_pandas_compatible(
        data=data, g=grouper.ContinuousGroup("fake_size", 10, label_type="median")
    )

    target = np.array(
        [
            -3.28356,
            -2.5578789,
            -1.8114079,
            -1.0532277,
            -0.3176548,
            0.4207354,
            1.1638769,
            1.907607,
            2.6944044,
            3.382559,
        ]
    )
    assert_almost_equal(pandas_group.cat.categories.to_numpy(), target)
