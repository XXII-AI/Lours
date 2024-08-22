from os.path import relpath
from pathlib import Path

import pandas as pd
import pytest

from libia.dataset import Dataset, from_caipy, from_coco, from_crowd_human
from libia.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_merge():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )

    n1, m1 = len(dataset1), dataset1.len_annot()
    n2, m2 = len(dataset2), dataset2.len_annot()

    with pytest.raises(ValueError):
        dataset1.merge(dataset2, allow_overlapping_image_ids=False)

    dataset = dataset1.merge(dataset2, ignore_index=True)
    assert (len(dataset)) == len(dataset1) + len(dataset2)
    assert (dataset.len_annot()) == dataset1.len_annot() + dataset2.len_annot()

    dataset_left = dataset1.merge(dataset2, ignore_index=True)
    dataset_right = dataset2.merge(dataset1, ignore_index=True)

    assert_dataset_equal(dataset_left, dataset_right, ignore_index=True)

    old_index = dataset2.images.index
    dataset2.images.index = pd.RangeIndex(n1, n1 + n2, name="id")
    dataset2.annotations.index = pd.RangeIndex(m1, m1 + m2, name="id")
    dataset2.annotations["image_id"] = dataset2.annotations["image_id"].replace(
        dict(zip(old_index, dataset2.images.index))
    )

    dataset_left = dataset1.merge(dataset2)
    dataset_right = dataset2.merge(dataset1)

    assert_dataset_equal(dataset_left, dataset_right)


def test_addition():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )

    assert_dataset_equal(dataset1 + dataset2, dataset2 + dataset1, ignore_index=True)

    assert_dataset_equal(
        dataset1 + dataset2,
        sum([dataset1, dataset2]),  # pyright: ignore
    )


def test_addition_incompatible_label_maps():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset").remap_from_preset(
        "coco", "supercategory"
    )
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )

    with pytest.warns(RuntimeWarning):
        dataset = dataset1 + dataset2

    assert_dataset_equal(
        dataset, dataset1.merge(dataset2, realign_label_map=True, ignore_index=True)
    )


def test_self_merge():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )

    merged = dataset + dataset

    assert len(dataset) == len(merged)
    assert merged.len_annot() == 2 * dataset.len_annot()


def test_merge_overlapping_ids():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    half1 = dataset.iloc_annot[::2]
    half2 = dataset.iloc_annot[1::2]
    merged_back = half1 + half2
    assert_dataset_equal(dataset, merged_back)


def test_merge_booleanize():
    tagged_dataset_path = DATA / "caipy_dataset" / "tags" / "small_tagged_dataset"
    dataset = from_caipy(dataset_path=tagged_dataset_path, use_schema=True)
    dataset1 = from_caipy(dataset_path=tagged_dataset_path, booleanize=False)[0]
    dataset2 = from_caipy(
        dataset_path=tagged_dataset_path,
        use_schema=True,
        booleanize=True,
    )[1]

    assert_dataset_equal(dataset, dataset1 + dataset2)

    dataset1 = from_caipy(dataset_path=tagged_dataset_path, use_schema=False)
    dataset1 = dataset1[0].booleanize("attributes.colors")
    dataset2 = from_caipy(
        dataset_path=tagged_dataset_path,
        use_schema=True,
        booleanize=True,
    )[1]

    assert_dataset_equal(dataset, dataset1 + dataset2)


def reconstruct_dataset_from_merged_marked(
    merged_dataset: Dataset, original_dataset: Dataset, check_image_origin: bool = False
) -> Dataset:
    dataset_back = merged_dataset.filter_annotations(
        merged_dataset.annotations["origin"] == original_dataset.dataset_name,
        remove_emptied_images=True,
    )
    if check_image_origin:
        assert (dataset_back.images["origin"] == original_dataset.dataset_name).all()
    dataset_back = (
        dataset_back.reset_images_root(original_dataset.images_root)
        .remap_classes(
            {k: k for k in original_dataset.label_map}, remove_not_mapped=True
        )
        .reset_index_from_mapping(
            images_index_map=dataset_back.images["origin_id"],
            annotations_index_map=dataset_back.annotations["origin_id"],
        )
    )
    back_images = dataset_back.images.astype(original_dataset.images.dtypes)
    back_images.index = back_images.index.astype(int)

    back_annotations = dataset_back.annotations.astype(
        original_dataset.annotations.dtypes
    )
    back_annotations.index = back_annotations.index.astype(int)
    return dataset_back.from_template(images=back_images, annotations=back_annotations)


def test_merge_mark_origin_disjoint_dataset():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )

    dataset = dataset1.merge(dataset2, mark_origin=True, ignore_index=True)
    dataset1_back = reconstruct_dataset_from_merged_marked(
        dataset, dataset1, check_image_origin=True
    )
    assert_dataset_equal(
        dataset1, dataset1_back, optional_columns=["origin", "origin_id"]
    )

    dataset2_back = reconstruct_dataset_from_merged_marked(
        dataset, dataset2, check_image_origin=True
    )
    assert_dataset_equal(
        dataset2, dataset2_back, optional_columns=["origin", "origin_id"]
    )


def test_merge_mark_origin_overlapping_dataset():
    dataset_og = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset1 = dataset_og.filter_annotations(
        slice(None, None, 2), mode="iloc", remove_emptied_images=True
    ).rename("A")
    dataset2 = dataset_og.filter_annotations(
        slice(1, None, 2), mode="iloc", remove_emptied_images=True
    ).rename("B")

    dataset = dataset1.merge(dataset2, mark_origin=True)
    dataset1_back = reconstruct_dataset_from_merged_marked(
        dataset, dataset1, check_image_origin=True
    )
    assert_dataset_equal(
        dataset1, dataset1_back, optional_columns=["origin", "origin_id"]
    )

    with pytest.raises(AssertionError):
        reconstruct_dataset_from_merged_marked(
            dataset, dataset2, check_image_origin=True
        )
    dataset2_back = reconstruct_dataset_from_merged_marked(
        dataset, dataset2, check_image_origin=False
    )
    assert_dataset_equal(
        dataset2, dataset2_back, optional_columns=["origin", "origin_id"]
    )


def test_merge_overwrite_origin():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )
    dataset3 = from_crowd_human(
        annotation_odgt=DATA / "crowdhuman_dataset/crowdhuman_train.odgt"
    ).remap_from_other(dataset1)

    dataset12 = dataset1.merge(dataset2, ignore_index=True)
    dataset_overwritten = dataset1.merge(
        dataset2, mark_origin=True, ignore_index=True
    ).merge(dataset3, mark_origin=True, ignore_index=True, overwrite_origin=True)

    dataset12_back = reconstruct_dataset_from_merged_marked(
        dataset_overwritten, dataset12
    )
    assert_dataset_equal(
        dataset12,
        dataset12_back,
        optional_columns=["origin", "origin_id"],
        remove_na_columns=True,
    )

    dataset = dataset1.merge(dataset2, mark_origin=True, ignore_index=True).merge(
        dataset3, mark_origin=True, ignore_index=True, overwrite_origin=False
    )

    dataset1_back = reconstruct_dataset_from_merged_marked(dataset, dataset1)
    assert_dataset_equal(
        dataset1,
        dataset1_back,
        optional_columns=["origin", "origin_id"],
        remove_na_columns=True,
    )
    dataset2_back = reconstruct_dataset_from_merged_marked(dataset, dataset2)
    assert_dataset_equal(
        dataset2,
        dataset2_back,
        optional_columns=["origin", "origin_id"],
        remove_na_columns=True,
    )
    dataset3_back = reconstruct_dataset_from_merged_marked(dataset, dataset3)
    assert_dataset_equal(
        dataset3,
        dataset3_back,
        optional_columns=["origin", "origin_id"],
        remove_na_columns=True,
    )


def test_images_root_absolute():
    dataset1 = from_caipy(dataset_path=DATA / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )
    dataset = dataset1 + dataset2
    assert dataset.images_root == DATA


def test_images_root_relpath():
    relative_data_path = Path(relpath(DATA))
    dataset1 = from_caipy(dataset_path=relative_data_path / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=relative_data_path / "coco_dataset/annotations_train.json",
        images_root=relative_data_path / "coco_dataset/data",
    )
    dataset = dataset1 + dataset2
    assert dataset.images_root == relative_data_path


def test_images_root_discrepancy():
    relative_data_path = Path(relpath(DATA))
    dataset1 = from_caipy(dataset_path=relative_data_path / "caipy_dataset")
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data",
    )
    dataset = dataset1 + dataset2
    assert dataset.images_root == DATA
