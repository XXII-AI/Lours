import pytest

from lours.utils.doc_utils import dummy_dataset
from lours.utils.testing import assert_dataset_equal


def test_dummy_dataset():
    first = dummy_dataset(
        n_imgs=10,
        n_annot=100,
        n_labels=2,
        n_list_columns_images=1,
        n_list_columns_annotations=1,
        booleanize="random",
        seed=1,
    )
    assert len(first) == 10
    assert first.len_annot() == 100

    second = dummy_dataset(
        n_imgs=10,
        n_annot=100,
        n_labels=2,
        n_list_columns_images=1,
        n_list_columns_annotations=1,
        booleanize="random",
        seed=1,
    )
    assert_dataset_equal(first, second)

    third = dummy_dataset(
        n_imgs=10,
        n_annot=100,
        n_labels=2,
        n_list_columns_images=1,
        n_list_columns_annotations=1,
        booleanize="random",
        seed=2,
    )
    with pytest.raises(AssertionError):
        assert_dataset_equal(first, third)


def test_dummy_dataset_bool():
    dataset = dummy_dataset(
        n_imgs=10,
        n_annot=100,
        n_labels=2,
        n_list_columns_images=1,
        n_list_columns_annotations=1,
        booleanize="all",
        seed=1,
    )
    assert len(dataset.booleanized_columns["images"]) == 1
    assert len(dataset.booleanized_columns["annotations"]) == 1


def test_dummy_dataset_real_images():
    dataset = dummy_dataset(
        n_imgs=10,
        n_annot=100,
        n_labels=2,
        seed=1,
        generate_real_images=True,
        n_list_columns_images=["my_list"],
    )
    dataset.check()
