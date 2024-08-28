from pathlib import Path

from numpy import dtype

from libia.dataset import from_caipy_generic
from libia.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_booleanize_dataset():
    dataset = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=True,
        booleanize=False,
    )

    to_booleanize = "attributes.colors"
    assert to_booleanize in dataset.annotations.columns
    assert dataset.annotations.dtypes[to_booleanize] == dtype("O")

    booleanized_dataset = dataset.booleanize(to_booleanize)
    assert "attributes.colors.white" in booleanized_dataset.annotations.columns
    assert "attributes.colors.blue" in booleanized_dataset.annotations.columns

    assert booleanized_dataset.annotations.dtypes["attributes.colors.white"] == dtype(
        "bool"
    )
    assert booleanized_dataset.annotations.dtypes["attributes.colors.blue"] == dtype(
        "bool"
    )

    debooleanized = booleanized_dataset.debooleanize()

    def sort_lists(x):
        if isinstance(x, list):
            return sorted(x)
        else:
            return x

    dataset.annotations[to_booleanize] = dataset.annotations[to_booleanize].apply(
        sort_lists
    )
    debooleanized.annotations[to_booleanize] = debooleanized.annotations[
        to_booleanize
    ].apply(sort_lists)
    assert_dataset_equal(dataset, debooleanized)
