import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ...utils import BBOX_COLUMN_NAMES
from ..dataset import Dataset
from .common import get_image_info, parse_annotation_name

BOX_XMIN, BOX_WIDTH, BOX_YMIN, BOX_HEIGHT = BBOX_COLUMN_NAMES


def from_crowd_human(
    annotation_odgt: Path | str,
    images_root: Path | str | None = None,
    visible_box: bool = True,
    dataset_name: str | None = None,
    split: str | None = None,
) -> Dataset:
    """Read a dataset in the format described for CrowdHuman

    See https://www.crowdhuman.org/download.html

    Args:
        annotation_odgt: annotation file using the format described in link given above.
        images_root: Folder where images are stored. Note that since annotations IDs are
            file names, all images must be at the root of this folder.
            If set to None, will use the image folder in
            the annotation file's parent folder. Defaults to None.
        visible_box: If set to True, will only take the visible bounding box. Otherwise,
            will take the whole bounding box, with possibly occluded parts.
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone. If not specified,
            the dataset name will be deduced from the name of the json file.
        split: Split value to give to the dataset. If set to None, will try to deduce
            it from the annotation file name. Defaults to None.

    Raises:
        FileNotFoundError: Will raise an error if the images_root folder does not exist
            Unlike COCO or Caipy, image size information is not stored in annotations
            and thus need to be computed by loading picture's headers.

    Returns:
        Dataset
    """
    annotation_odgt = Path(annotation_odgt)
    if images_root is None:
        images_root = annotation_odgt.parent / "images"
    else:
        images_root = Path(images_root)

    if not images_root.is_dir():
        raise FileNotFoundError(
            "Image folder needs to exist as CrowdHuman's annotations don't have image"
            " size information"
        )

    parsed_dataset_name, parsed_split = parse_annotation_name(
        annotations_file_path=annotation_odgt
    )
    if dataset_name is None:
        dataset_name = parsed_dataset_name
    if split is None:
        split = parsed_split

    with open(annotation_odgt) as f:
        lines = f.read().strip().split("\n")

    annots = [json.loads(line) for line in lines]
    image_dicts = []
    annotation_dicts = []
    label_map = {0: "person", 1: "head"}
    ids_map = {v: k for k, v in label_map.items()}
    for image_id, image_data in enumerate(tqdm(annots)):
        relative_path = Path(f"{image_data['ID']}.jpg")
        image_path = images_root / relative_path
        image_info = get_image_info(image_id, relative_path, image_path)
        image_dicts.append(image_info)

        current_annotations = []
        if "gtboxes" not in image_data:
            continue
        for annot in image_data["gtboxes"]:
            if annot["tag"] == "mask":
                continue
            if "ignore" in annot["extra"] and annot["extra"]["ignore"] == 1:
                continue

            *_, box_w, box_h = annot["fbox"]
            area = box_w * box_h
            *_, vbox_w, vbox_h = annot["vbox"]
            visible_share = vbox_w * vbox_h / area
            person_annotation = {
                **annot["extra"],
                "image_id": image_id,
                "category_id": ids_map["person"],
                "bbox": annot["vbox"] if visible_box else annot["fbox"],
                "visible_share": visible_share,
            }
            current_annotations.append(person_annotation)
            if "ignore" in annot["head_attr"] and annot["head_attr"]["ignore"] == 1:
                continue
            # We don't know how visible each head is
            # Sometimes it's occluded, sometimes it's even out of the screen
            head_annotation = {
                **annot["extra"],
                **annot["head_attr"],
                "image_id": image_id,
                "category_id": ids_map["head"],
                "bbox": annot["hbox"],
                "visible_share": float("nan"),
            }
            current_annotations.append(head_annotation)
        annotation_dicts.extend(current_annotations)
    images = pd.DataFrame.from_records(image_dicts).set_index("id")
    images["split"] = split

    annotations = pd.DataFrame.from_records(annotation_dicts)
    annotations.index.name = "id"
    bboxes = pd.DataFrame(list(annotations["bbox"]), index=annotations.index)
    for i, name in enumerate(BBOX_COLUMN_NAMES):
        annotations[name] = bboxes[i]
    annotations["split"] = split
    annotations = annotations.drop(["bbox", "ignore"], axis=1, errors="ignore")
    annotations["unsure"] = annotations["unsure"].fillna(0)

    return Dataset(
        images_root=images_root,
        images=images,
        annotations=annotations,
        label_map=label_map,
        dataset_name=dataset_name,
    ).cap_bounding_box_coordinates()
