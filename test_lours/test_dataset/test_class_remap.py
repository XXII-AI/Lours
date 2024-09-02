from pathlib import Path

import pandas as pd
import pytest

from lours.dataset import from_coco
from lours.utils.label_map_merger import merge_label_maps

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_class_remap_from_dict():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    assert len(dataset) == 2
    assert len(dataset.label_map) == 80
    assert dataset.len_annot() == 21

    class_mapping = {
        1: 1,
        62: 3,
        64: 3,
        67: 3,
        72: 2,
        78: 2,
        82: 2,
        84: 1,
        85: 3,
        86: 3,
    }
    remapped_dataset = dataset.remap_classes(
        class_mapping=class_mapping,
        new_names={1: "person", 2: "electronics", 3: "object"},
    )
    assert len(remapped_dataset.label_map) == 3
    assert remapped_dataset.len_annot() == 20

    remapped_dataset = dataset.remap_classes(
        class_mapping=class_mapping,
        new_names={1: "person", 2: "electronics", 3: "object"},
        remove_emptied_images=True,
    )
    assert len(remapped_dataset.label_map) == 3
    assert remapped_dataset.len_annot() == 20
    assert len(remapped_dataset) == 1

    remapped_dataset = dataset.remap_classes(
        class_mapping=class_mapping,
        new_names={1: "person", 2: "electronics", 3: "object"},
        remove_not_mapped=False,
    )
    assert len(remapped_dataset.label_map) == 71
    assert remapped_dataset.len_annot() == 21


def test_class_remap_from_preset():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    assert len(dataset.label_map) == 80
    assert dataset.len_annot() == 21
    assert len(dataset) == 2

    with pytest.raises(ValueError):
        dataset.remap_from_preset("coco", "pascal")

    remapped_dataset = dataset.remap_from_preset("coco", "pascalvoc")

    assert len(remapped_dataset.label_map) == 20
    assert remapped_dataset.len_annot() == 11

    remapped_dataset = dataset.remap_from_preset(
        "coco", "pascalvoc", remove_not_mapped=False
    )
    assert len(remapped_dataset.label_map) == 74
    assert remapped_dataset.len_annot() == 21

    remapped_dataset = dataset.remap_from_preset(
        "coco", "pascalvoc", remove_emptied_images=True
    )
    assert len(remapped_dataset) == 1


def test_class_remap_from_dataframe():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    assert len(dataset.label_map) == 80
    assert dataset.len_annot() == 21
    assert len(dataset) == 2

    remap_df = pd.DataFrame(
        [
            [1, "person", 1, "person"],
            [62, "chair", 3, "object"],
            [64, "potted plant", 3, "object"],
            [67, "dining table", 3, "object"],
            [72, "tv", 2, "electronics"],
            [78, "microwave", 2, "electronics"],
            [82, "refrigerator", 2, "electronics"],
            [84, "book", 1, "object"],
            [85, "clock", 3, "object"],
            [86, "vase", 3, "object"],
        ],
        columns=[
            "input_category_id",
            "input_category_name",
            "output_category_id",
            "output_category_name",
        ],
    )
    remapped_dataset = dataset.remap_from_dataframe(remap_df)
    assert len(remapped_dataset.label_map) == 3
    assert remapped_dataset.len_annot() == 20

    remapped_dataset = dataset.remap_from_dataframe(remap_df, remove_not_mapped=False)
    assert len(remapped_dataset.label_map) == 71
    assert remapped_dataset.len_annot() == 21

    remapped_dataset = dataset.remap_from_dataframe(
        remap_df, remove_emptied_images=True
    )
    assert len(remapped_dataset.label_map) == 3
    assert len(remapped_dataset) == 1


def test_class_remap_from_csv():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )
    assert len(dataset.label_map) == 80
    assert dataset.len_annot() == 21
    assert len(dataset) == 2

    csv_file = DATA / "remap_dataset/remap.csv"
    remapped_dataset = dataset.remap_from_csv(csv_file)
    assert len(remapped_dataset.label_map) == 3
    assert remapped_dataset.len_annot() == 20

    remapped_dataset = dataset.remap_from_csv(csv_file, remove_not_mapped=False)
    assert len(remapped_dataset.label_map) == 71
    assert remapped_dataset.len_annot() == 21

    remapped_dataset = dataset.remap_from_csv(csv_file, remove_emptied_images=True)
    assert len(remapped_dataset.label_map) == 3
    assert remapped_dataset.len_annot() == 20
    assert len(remapped_dataset) == 1


def test_class_remap_from_other():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data",
        split="valid",
    )

    assert len(dataset.label_map) == 80
    assert dataset.len_annot() == 21
    assert len(dataset) == 2

    other_dataset = dataset.remap_from_preset("coco", "pascalvoc")

    remapped_dataset = dataset.remap_from_other(other_dataset)

    assert len(remapped_dataset.label_map) == 80

    merged_label_map = merge_label_maps(
        remapped_dataset.label_map, other_dataset.label_map, method="outer"
    )

    assert len(merged_label_map) == 86
