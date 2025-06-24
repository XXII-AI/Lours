import pytest

from lours.utils.doc_utils import dummy_dataset, set_attribute_columns_labels
from lours.utils.testing import assert_dataset_equal
import pandas as pd
import numpy as np
from faker import Faker


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


def test_column_specs(dataframe_regression):
    gen = np.random.default_rng(1)
    Faker.seed(seed=1)
    length = 1000
    fake_generator = Faker()
    dataframe_to_test = pd.DataFrame([], index=range(length))
    set_attribute_columns_labels(
        dataframe_to_test,
        columns_specs=2,
        fake_generator=fake_generator,
        numpy_generator=gen,
    )
    set_attribute_columns_labels(
        dataframe_to_test,
        columns_specs=["col1", "col2"],
        fake_generator=fake_generator,
        numpy_generator=gen,
    )
    set_attribute_columns_labels(
        dataframe_to_test,
        columns_specs=[
            ["value1", "value2"],
            [0.5, 0.5],
            {"value3": 0.1, "value4": 0.9},
        ],
        fake_generator=fake_generator,
        numpy_generator=gen,
    )
    set_attribute_columns_labels(
        dataframe_to_test,
        columns_specs={
            "col3": 1,
            "col4": 2,
            "col5": [0.5, 0.5],
            "col6": ["value5", "value6"],
            "col7": {"value7": 0.1, "value8": 0.9},
        },
        fake_generator=fake_generator,
        numpy_generator=gen,
    )

    dataframe_regression.check(dataframe_to_test)

    np.testing.assert_allclose(
        dataframe_to_test["col5"].value_counts().to_numpy() / length,
        [0.5, 0.5],
        atol=0.05,
    )
    pd.testing.assert_series_equal(
        dataframe_to_test["col7"].value_counts() / length,
        pd.Series(
            [0.1, 0.9], index=pd.Index(["value7", "value8"], name="col7"), name="count"
        ),
        check_categorical=False,
        check_index_type=False,
        check_like=True,
        atol=0.05,
    )
