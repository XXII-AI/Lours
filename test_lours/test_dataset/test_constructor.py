from pathlib import Path

import pytest
from pathlib import Path

from lours.utils.doc_utils import dummy_dataset
from lours.utils.testing import assert_dataset_equal
from lours.dataset import Dataset


def test_empty_constructor():
    my_dataset = Dataset()
    assert len(my_dataset) == 0
    assert my_dataset.len_annot() == 0
    assert set(my_dataset.images.columns) == {
        "width",
        "height",
        "relative_path",
        "type",
    }
    assert set(my_dataset.annotations.columns) == {
        "image_id",
        "category_id",
        "category_str",
        "box_x_min",
        "box_y_min",
        "box_width",
        "box_height",
    }
    assert_dataset_equal(my_dataset, my_dataset.get_split(None))
    assert my_dataset.label_map == {}


def test_missing_category_id():
    my_dummy_dataset = dummy_dataset(label_map={0: "0"})
    my_dummy_dataset.annotations.loc[
        my_dummy_dataset.annotations.index[0], "category_id"
    ] = 1
    with pytest.warns(RuntimeWarning):
        my_dataset = Dataset(
            images_root=Path("."),
            images=my_dummy_dataset.images,
            annotations=my_dummy_dataset.annotations,
            label_map={0: "0"},
        )
    assert my_dataset.label_map == {0: "0", 1: "1"}
