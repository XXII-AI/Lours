from pathlib import Path

from lours.dataset import from_caipy, from_coco
from lours.utils.difftools import dataset_diff
from lours.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_diff_same_dataset():
    dataset = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )

    dataset2 = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )
    left, right, common = dataset_diff(dataset, dataset2)
    assert len(left) == 0
    assert len(right) == 0
    assert_dataset_equal(dataset, common)


def test_diff_different_annotation():
    dataset = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )

    dataset2 = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )
    dataset2.annotations.loc[dataset.annotations.index[0], "box_x_min"] = 0

    left, right, common = dataset_diff(dataset, dataset2)

    assert left.len_annot() == 1 and left.annotations.iloc[0].equals(
        dataset.annotations.iloc[0]
    )
    assert right.len_annot() == 1 and right.annotations.iloc[0].equals(
        dataset2.annotations.iloc[0]
    )

    assert common.len_annot() == dataset.len_annot() - 1

    left2, right2, common2 = dataset2 - dataset
    assert_dataset_equal(left, right2)
    assert_dataset_equal(right, left2)
    assert_dataset_equal(common, common2)


def test_diff_additional_image():
    dataset = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )

    dataset2 = from_caipy(
        dataset_path=DATA / "caipy_dataset",
    )

    dataset2 = dataset2.loc[dataset2.images.index[1:]]

    left, right, common = dataset_diff(dataset, dataset2)

    assert_dataset_equal(left, dataset.iloc[[0]])
    assert len(right) == 0
    assert_dataset_equal(common, dataset2)
    assert_dataset_equal(left + common, dataset)

    left, right, common = dataset_diff(dataset2, dataset)

    assert_dataset_equal(right, dataset.iloc[[0]])
    assert len(left) == 0
    assert_dataset_equal(common, dataset2)
    assert_dataset_equal(right + common, dataset)


def test_diff_additional_annotations():
    dataset = from_coco(coco_json=DATA / "coco_dataset/annotations_valid.json")
    dataset2 = from_coco(coco_json=DATA / "coco_dataset/annotations_valid.json")

    dataset2 = dataset2.loc_annot[dataset2.annotations.index[:-1]]

    left, right, common = dataset_diff(dataset, dataset2)

    assert_dataset_equal(left, dataset.iloc_annot[-1].remove_empty_images())
    assert len(right) == 0
    assert_dataset_equal(common, dataset2)
    assert_dataset_equal(left + common, dataset)

    left, right, common = dataset_diff(dataset2, dataset)

    assert_dataset_equal(right, dataset.iloc_annot[-1].remove_empty_images())
    assert len(left) == 0
    assert_dataset_equal(common, dataset2)
    assert_dataset_equal(right + common, dataset)


def test_diff_exclude_columns():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
    )
    dataset2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
    )

    dataset.annotations["A"] = False
    dataset.images["B"] = 0

    left, right, common = dataset_diff(
        dataset, dataset2, exclude_annotations_columns="A", exclude_image_columns="B"
    )

    assert len(left) == 0
    assert len(right) == 0
    assert_dataset_equal(common, dataset2)

    dataset2.annotations["C"] = "a"
    dataset2.images["D"] = None

    left, right, common = dataset_diff(
        dataset,
        dataset2,
        exclude_annotations_columns=["A", "C"],
        exclude_image_columns=["B", "D"],
    )

    original = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
    )

    assert len(left) == 0
    assert len(right) == 0
    assert_dataset_equal(common, original)


def test_diff_full():
    full_dataset = from_coco(
        coco_json=DATA / "coco_eval" / "instances_val2017.json",
    ).iloc[:10]
    full_dataset.images = full_dataset.images.drop(columns=["coco_url", "flickr_url"])
    full_dataset.annotations = full_dataset.annotations.drop(
        columns=["segmentation.counts", "segmentation.size"]
    )
    full_dataset.images["test_none"] = None
    full_dataset.annotations["other_test_none"] = None
    # Remove first image from first dataset
    # Also, remove first annotation
    image_index = full_dataset.images.index
    image_to_remove_from_left = image_index[0]
    image_to_remove_from_right = image_index[1]

    annotation_to_remove_from_left = full_dataset.iloc[2].annotations.index[0]
    annotation_to_remove_from_right = full_dataset.iloc[3].annotations.index[0]
    image_to_modify_left = image_index[4]
    image_to_modify_right = image_index[5]
    annotation_to_modify_left = full_dataset.iloc[6].annotations.index[0]
    annotation_to_modify_right = full_dataset.iloc[7].annotations.index[0]

    dataset_left = full_dataset.loc[
        full_dataset.images.index != image_to_remove_from_left
    ]
    dataset_left = dataset_left.loc_annot[
        dataset_left.annotations.index != annotation_to_remove_from_left
    ]
    dataset_left.images.loc[image_to_modify_left, "width"] = 0
    dataset_left.annotations.loc[annotation_to_modify_left, "box_width"] = 0

    dataset_right = full_dataset.loc[
        full_dataset.images.index != image_to_remove_from_right
    ]
    dataset_right = dataset_right.loc_annot[
        dataset_right.annotations.index != annotation_to_remove_from_right
    ]
    dataset_right.images.loc[image_to_modify_right, "width"] = 0
    dataset_right.annotations.loc[annotation_to_modify_right, "box_width"] = 0

    left_diff, right_diff, common = dataset_diff(dataset_left, dataset_right)
    assert_dataset_equal(
        left_diff,
        dataset_left.loc[
            [image_to_remove_from_right, image_to_modify_left, image_to_modify_right]
        ]
        + dataset_left.loc_annot[
            [
                annotation_to_remove_from_right,
                annotation_to_modify_left,
                annotation_to_modify_right,
            ]
        ].remove_empty_images(),
    )
    assert_dataset_equal(
        right_diff,
        dataset_right.loc[
            [image_to_remove_from_left, image_to_modify_left, image_to_modify_right]
        ]
        + dataset_right.loc_annot[
            [
                annotation_to_remove_from_left,
                annotation_to_modify_left,
                annotation_to_modify_right,
            ]
        ].remove_empty_images(),
    )

    assert_dataset_equal(dataset_left, common + left_diff)

    assert_dataset_equal(dataset_right, common + right_diff)
