from pathlib import Path

from pandas.testing import assert_frame_equal, assert_series_equal

from libia.dataset import from_coco

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_one_frame():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco_single_image = coco[0]

    single_image, annotations = coco.get_one_frame(0)

    assert_series_equal(coco_single_image.images.iloc[0], single_image)
    assert_frame_equal(coco_single_image.annotations, annotations)


def test_empty():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    empty_coco = coco.empty_annotations()

    assert_frame_equal(coco.images, empty_coco.images)
    assert set(coco.annotations.columns) == set(empty_coco.annotations.columns)
    assert empty_coco.len_annot() == 0


def test_subsampling():
    coco = from_coco(
        coco_json=DATA / "coco_eval/instances_val2017.json", images_root=Path(".")
    )
    coco_single_image = coco[::10]

    subsampled_images = coco.images.iloc[::10]
    subsampled_annotations = coco.annotations[
        coco.annotations["image_id"].isin(subsampled_images.index)
    ]

    assert_frame_equal(coco_single_image.images, subsampled_images)
    assert_frame_equal(coco_single_image.annotations, subsampled_annotations)
