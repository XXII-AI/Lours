from pathlib import Path

import pandas as pd

from lours.dataset import Dataset, from_caipy
from lours.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_indexing():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    subset1 = dataset[:2]
    subset2 = dataset[2:]
    assert_dataset_equal(dataset, subset1 + subset2)

    subset1 = dataset[::2]
    subset2 = dataset[1::2]
    assert_dataset_equal(dataset, subset1 + subset2)


def test_iter_images():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    result = list(dataset.iter_images())
    images = pd.DataFrame.from_records([r[0] for r in result])
    images.index = pd.Index([r[0].name for r in result], name="id")
    annotations = pd.concat([r[1] for r in result])
    dataset2 = dataset.from_template(images=images, annotations=annotations)
    assert_dataset_equal(dataset, dataset2)


def test_iter_split():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    result = list(dataset.iter_splits())
    assert len(result) == 2
    for split_name, split in result:
        assert len(split.images["split"].unique()) == 1
        assert split_name == split.images["split"].unique()[0]
    dataset2 = sum(
        [r[1] for r in result], start=Dataset(images_root=dataset.images_root)
    )
    assert_dataset_equal(dataset, dataset2)
