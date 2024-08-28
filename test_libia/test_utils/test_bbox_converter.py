from pathlib import Path

import numpy as np

from libia.dataset import from_coco
from libia.utils import BBOX_COLUMN_NAMES, bbox_converter

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_bbox_column_names():
    assert BBOX_COLUMN_NAMES == ["box_x_min", "box_y_min", "box_width", "box_height"]


def test_convert_bbox():
    coco = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )
    bbox = coco.annotations[
        ["image_id", "box_x_min", "box_width", "box_y_min", "box_height"]
    ]

    target = (
        bbox.to_numpy()[:, 1:]
        / coco.images.loc[
            bbox["image_id"], ["width", "width", "height", "height"]
        ].to_numpy()
    )
    target[:, 1] += target[:, 0]
    target[:, 3] += target[:, 2]
    converted = bbox_converter.convert_bbox(
        bbox, images_df=coco.images, input_format="XWYH", output_format="xxyy"
    ).to_numpy()

    np.testing.assert_almost_equal(target, converted)

    target = (
        bbox.to_numpy()[:, [1, 3, 2, 4]]  # pyright: ignore
        / coco.images.loc[
            coco.annotations["image_id"], ["width", "height", "width", "height"]
        ].to_numpy()
    )
    converted = bbox_converter.convert_bbox(
        bbox, images_df=coco.images, input_format="XWYH", output_format="xywh"
    ).to_numpy()

    np.testing.assert_almost_equal(target, converted)


def test_export_bbox():
    coco = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    converted = bbox_converter.export_bbox(
        coco.annotations, coco.images, output_format="cxwcyh"
    ).to_numpy()
    target = coco.annotations[
        ["box_x_min", "box_width", "box_y_min", "box_height"]
    ].to_numpy()
    target /= coco.images.loc[
        coco.annotations["image_id"], ["width", "width", "height", "height"]
    ].to_numpy()
    target[:, 0] += target[:, 1] / 2
    target[:, 2] += target[:, 3] / 2
    np.testing.assert_almost_equal(target, converted)

    converted = bbox_converter.export_bbox(
        coco.annotations, coco.images, output_format="cxcywh"
    ).to_numpy()
    target = coco.annotations[
        ["box_x_min", "box_y_min", "box_width", "box_height"]
    ].to_numpy()
    target /= coco.images.loc[
        coco.annotations["image_id"], ["width", "height", "width", "height"]
    ].to_numpy()
    target[:, 0] += target[:, 2] / 2
    target[:, 1] += target[:, 3] / 2
    np.testing.assert_almost_equal(target, converted)

    converted = bbox_converter.export_bbox(
        coco.annotations, coco.images, output_format="XXYY"
    ).to_numpy()
    target = coco.annotations[
        ["box_x_min", "box_width", "box_y_min", "box_height"]
    ].to_numpy()
    target[:, 1] += target[:, 0]
    target[:, 3] += target[:, 2]
    np.testing.assert_almost_equal(target, converted)

    converted = bbox_converter.export_bbox(
        coco.annotations, coco.images, output_format="XYXY"
    ).to_numpy()
    target = coco.annotations[
        ["box_x_min", "box_y_min", "box_width", "box_height"]
    ].to_numpy()
    target[:, 2] += target[:, 0]
    target[:, 3] += target[:, 1]
    np.testing.assert_almost_equal(target, converted)
