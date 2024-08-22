from pathlib import Path

from libia.dataset import from_coco
from libia.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


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

    left_diff, right_diff, common = dataset_left - dataset_right
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

    left_diff2, right_diff2, common2 = dataset_right - dataset_left

    assert_dataset_equal(left_diff, right_diff2)
    assert_dataset_equal(right_diff, left_diff2)
    assert_dataset_equal(common, common2)
