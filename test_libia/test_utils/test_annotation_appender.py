from pathlib import Path

import numpy as np
import pytest

from libia.dataset import from_coco
from libia.utils.testing import assert_dataset_equal

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_add_annotations():
    coco = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )
    coco = coco.add_detection_annotation(
        285, [0, 10, 20, 30], 18, "XXYY", new_attribute=0
    )
    assert coco.len_annot() == 22
    coco = coco.add_detection_annotation(
        285, [[0, 10, 20, 30], [10, 12, 30, 32]], 18, "XXYY", new_attribute=0
    )
    assert coco.len_annot() == 24
    assert "new_attribute" in coco.annotations.columns
    assert len(coco.annotations["new_attribute"].dropna()) == 3

    coco = coco.add_detection_annotation(
        image_id=np.array([285, 285]),
        bbox_coordinates=np.array([[0, 10, 20, 30], [10, 12, 30, 32]]),
        category_id=np.array([18, 18]),
        format_string="XXYY",
    )
    assert coco.len_annot() == 26

    with pytest.raises(ValueError):
        coco.add_detection_annotation(
            image_id=np.array([285, 285, 139]),
            bbox_coordinates=np.array([[0, 10, 20, 30], [10, 12, 30, 32]]),
            category_id=np.array([18, 18]),
            format_string="XXYY",
        )


def test_add_appender():
    coco1 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    with coco1.annotation_append("XXYY") as aa:
        aa.append(
            285, np.array([[0, 10, 20, 30], [10, 12, 30, 32]]), np.array([18, 18])
        )
        aa.append(
            139, np.array([[0, 10, 20, 30], [10, 12, 30, 32]]), np.array([18, 18])
        )

    assert coco1.len_annot() == 25

    coco2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    appender = coco2.annotation_append("XXYY")
    appender.append(
        np.array([285, 285, 139, 139]),
        np.array(
            [
                [0, 10, 20, 30],
                [10, 12, 30, 32],
                [0, 10, 20, 30],
                [10, 12, 30, 32],
            ]
        ),
        np.array([18, 18, 18, 18]),
    )
    appender.finish()

    assert_dataset_equal(coco1, coco2)

    coco3 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    with coco3.annotation_append("XXYY", category_ids_mapping={2: 18}) as aa:
        aa.append(285, np.array([[0, 10, 20, 30], [10, 12, 30, 32]]), 2)
        aa.append(139, np.array([[0, 10, 20, 30], [10, 12, 30, 32]]), np.array([2, 2]))

    assert_dataset_equal(coco1, coco3)


def test_append_nothing():
    coco1 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    with coco1.annotation_append("XXYY"):
        pass

    coco2 = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    assert_dataset_equal(coco1, coco2)


def test_append_various_attributes():
    coco = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    with coco.annotation_append("XXYY", label_map={333: "new class"}) as aa:
        aa.append(
            139,
            attribute_A="hello",
            bbox_coordinates=np.array([[0, 10, 20, 30], [10, 12, 30, 32]]),
            category_id=333,
        )
        aa.append(
            139,
            attribute_B=["hi", "hi"],
            bbox_coordinates=np.array([[0, 10, 20, 30], [10, 12, 30, 32]]),
            category_id=np.array([2, 2]),
        )

    # Assert that non NaN values of attribute_A and attribute_B are not on the same
    # index, since it was not on the same append call
    assert (
        set(coco.annotations["attribute_A"].dropna().index).intersection(
            set(coco.annotations["attribute_B"].dropna().index)
        )
        == set()
    )

    with coco.annotation_append() as aa, pytest.raises(ValueError):
        aa.append(333, bbox_coordinates=[0, 10, 20, 30], category_id=2)
