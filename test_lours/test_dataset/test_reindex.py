from pathlib import Path

import pandas as pd

from lours.dataset import from_coco
from lours.utils.testing import assert_dataset_equal, assert_ids_well_formed

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_match_index():
    dataset1 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
        split="train",
    )
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train_reindex.json",
        images_root=DATA / "coco_dataset/data",
        split="train",
    )

    reindex_dataset2 = dataset2.match_index(dataset1.images)

    pd.testing.assert_frame_equal(
        reindex_dataset2.images, dataset1.images, check_like=True
    )
    assert_ids_well_formed(reindex_dataset2)


def test_match_index_subset():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
        split="train",
    )
    modified_images = dataset.images.iloc[::2].reset_index(drop=True)

    remapped_dataset = dataset.match_index(modified_images, remove_unmatched=False)

    assert_ids_well_formed(remapped_dataset)
    assert_dataset_equal(dataset.reset_index(), remapped_dataset.reset_index())

    remapped_dataset2 = dataset.match_index(modified_images, remove_unmatched=True)

    assert_ids_well_formed(remapped_dataset2)
    assert len(remapped_dataset2) == len(modified_images)
    assert_dataset_equal(
        dataset.iloc[::2].reset_index(), remapped_dataset2.reset_index()
    )


def test_reset_index_mapping():
    dataset = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json",
    )

    images_index_map = pd.Series([3, 2, 1], index=dataset.images.index[:3])
    annotations_index_map = pd.Series([3, 2, 1], index=dataset.annotations.index[:3])

    remapped_dataset = dataset.reset_index_from_mapping(
        images_index_map=images_index_map,
        annotations_index_map=annotations_index_map,
        remove_unmapped=True,
    )
    assert len(remapped_dataset) == 3

    assert_ids_well_formed(remapped_dataset)

    remapped_dataset2 = dataset.reset_index_from_mapping(
        images_index_map=images_index_map,
        annotations_index_map=annotations_index_map,
        remove_unmapped=False,
    )

    assert len(remapped_dataset2) == len(dataset)
    assert remapped_dataset2.len_annot() == dataset.len_annot()

    assert_ids_well_formed(remapped_dataset2)
