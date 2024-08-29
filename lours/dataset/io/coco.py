import json
import shutil
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
from imageio.v3 import imread, imwrite

from lours.utils import BBOX_COLUMN_NAMES
from lours.utils.bbox_converter import (
    column_names_from_format_string,
    export_bbox,
    import_bbox,
)
from lours.utils.testing import assert_images_valid

from ..dataset import Dataset
from .common import parse_annotation_name


def from_coco(
    coco_json: Path | str,
    images_root: Path | str | None = None,
    dataset_name: str | None = None,
    split: str | None = None,
    label_map: dict[int, str] | None = None,
    box_format: str = "XYWH",
    drop_columns: Iterable[str] = ("iscrowd", "segmentation"),
) -> Dataset:
    """Load a coco json file into a dictionary.
    Note that there is only one split per file, which needs to be given by caller.
    See `specifications`__ (only Object detection)

        .. __: https://cocodataset.org/#format-data

    Notes:
        - ``from_coco`` is compatible with bounding box annotations without
          ``category_id`` field, but then you will need to have a label map of only one
          entry, which will be assigned to every bounding box.
        - If split value is not given, it will try to deduce it from the file name.
          More specifically, it will search a ``<name>_<split>.json`` pattern and assign
          ``name`` to the dataset name and ``split`` to the split value.

    Args:
        coco_json: path of json file
        images_root: folder which file_name of images are relative to
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone. If not specified,
            the dataset name will be deduced from the name of the json file.
        split: split of given json file.
            If not set, will try to deduce from filename. Defaults to None.
        label_map: Optional dictionary to specify the name of each category id. If not
            set, will try to deduce it from the json itself, in the field `categories`
            at its root.
        box_format: what type of annotation the json file will have.
            It will be converted back to XYWH. Defaults to XYWH
        drop_columns: list of names of columns that need to be dropped from the parsed
            json dictionary.

    Returns:
        Loaded dataset object
    """
    # If given paths are string, convert them to Path
    coco_json = Path(coco_json)
    if images_root is not None:
        images_root = Path(images_root)
    else:
        images_root = coco_json.parent
    parsed_dataset_name, parsed_split = parse_annotation_name(
        annotations_file_path=coco_json
    )
    if dataset_name is None:
        dataset_name = parsed_dataset_name
    if split is None:
        split = parsed_split
    with open(coco_json) as f:
        coco_annotations = json.load(f)

    images = pd.json_normalize(coco_annotations["images"]).set_index("id")
    images["relative_path"] = images["file_name"].apply(Path)  # pyright: ignore
    images["type"] = images["relative_path"].apply(lambda x: x.suffix)
    if split is not None:
        images["split"] = split
    images = images.drop("file_name", axis=1)

    if label_map is None:
        try:
            label_map = {c["id"]: c["name"] for c in coco_annotations["categories"]}
        except KeyError:
            label_map = {}

    try:
        annotations = pd.json_normalize(coco_annotations["annotations"]).set_index("id")
    except KeyError:
        annotations = pd.DataFrame(
            [],
            columns=[
                *column_names_from_format_string("XYWH"),
                "image_id",
                "category_id",
            ],
        )
        annotations.index.name = "id"
        return Dataset(images_root, images, annotations, label_map)
    # Don't deal with iscrowd=1 for now
    if "iscrowd" in annotations.columns:
        annotations = annotations[annotations["iscrowd"] == 0]
    bboxes = pd.DataFrame(
        list(annotations["bbox"]),
        index=annotations.index,
        columns=column_names_from_format_string(box_format),
    )
    bboxes = import_bbox(
        bboxes, images, image_ids=annotations["image_id"], input_format=box_format
    )
    annotations = pd.concat([annotations, bboxes], axis=1)
    annotations = annotations.drop([*drop_columns, "bbox"], axis=1, errors="ignore")

    if "category_id" not in annotations.columns:
        assert len(label_map) == 1
        annotations["category_id"] = list(label_map)[0]

    if annotations["category_id"].hasnans:
        raise ValueError(
            "Some category ids in annotations are undefined. Make sure either every"
            " annotation in your coco file has a `category_id` field, or none of them"
            " have"
        )

    if "score" in annotations:
        annotations = annotations.rename(columns={"score": "confidence"})
    return Dataset(images_root, images, annotations, label_map, dataset_name)


def from_coco_keypoints(
    coco_json: Path | str,
    images_root: Path | str | None = None,
    dataset_name: str | None = None,
    split: str | None = None,
    box_format: str = "XY",
    category_name: str | None = "head",
):
    """Special coco loading function for crowds, it will assume point box format
    (either XY or xy), only one category, with an id of 0, and a category name of person
    (unless specified otherwise in the coco format)

    Args:
        coco_json: path of json file
        images_root: folder which file_name of images are relative to
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone. If not specified,
            the dataset name will be deduced from the name of the json file.
        split: split of given json file.
            If not set, will try to deduce from filename. Defaults to None.
        box_format: what type of annotation the json file will have.
            It will be converted back to XYWH, with box width and height set to 0.
            Defaults to XY
        category_name: name of the only category of this coco json file.
            It will then call the ``from_coco`` original version with a label map option
            set to ``{0: category_name}``.
            If set to None, will deduce it from coco file. Defaults to "person".

    Returns:
        Loaded dataset object
    """
    return from_coco(
        coco_json=coco_json,
        images_root=images_root,
        dataset_name=dataset_name,
        split=split,
        box_format=box_format,
        label_map={0: category_name} if category_name is not None else None,
    )


def dataset_to_coco(
    dataset: Dataset,
    output_path: Path | str,
    copy_images: bool = False,
    to_jpg: bool = True,
    overwrite_images: bool = False,
    overwrite_labels: bool = False,
    add_split_suffix: bool | None = None,
    box_format: str = "XYWH",
    version: str = "0",
    contributor: str = "XXII",
) -> None:
    """Save dataset to COCO format. Note that by default, no image or image path is
    manipulated

    Args:
        dataset: Dataset object to save
        output_path: output folder where to save the json file,
            and optionally the images. Can also be the name of the output json file when
            there is only one split value.
        copy_images: If True, will copy images linked by annotations in
            a "data" folder, similar to 51. Defaults to False.
        to_jpg: if True, along with previous option, will convert
            images to jpg if needed. Defaults to True.
        overwrite_images: if False with copy_images True,
            will skip images that are already copied. Defaults to True.
        overwrite_labels: if False,
            will skip annotation that are already created. Defaults to True.
        add_split_suffix: if True, will add the split name to the
            name of the json file. cannot be False if dataset has multiple splits.
            If not set, will add suffix only if dataset has multiple splits.
        box_format: what type of annotation the json file will have.
            It will be converted from XYWH. Defaults to XYWH
        version: Arbitrary version number for dataset metadata.
            Defaults to "0".
        contributor: Arbitrary contributor info for dataset metadata.
            Defaults to "XXII".
    """
    if copy_images:
        try:
            assert_images_valid(dataset, load_images=False)
        except AssertionError as e:
            raise ValueError(
                "Dataset images are missing, check that the images root folder is the"
                " right one"
            ) from e
    output_path = Path(output_path).resolve()
    dataset = dataset.debooleanize()
    if output_path.suffix == ".json":
        output_file_name = output_path.stem
        output_path = output_path.parent
    else:
        output_path.mkdir(exist_ok=True, parents=True)
        output_file_name = "annotations"
    now = date.today()

    if add_split_suffix is None:
        add_split_suffix = len(dataset.annotations.split.unique()) > 1

    if not add_split_suffix:
        assert (
            "split" not in dataset.annotations
            or len(dataset.annotations.split.unique()) == 1
        ), "Cannot remove split suffix because dataset has more than one split"

    if copy_images:
        output_img_paths = dataset.images["relative_path"].apply(
            lambda x: output_path / "data" / x
        )
        if to_jpg:
            output_img_paths = output_img_paths.apply(lambda x: x.with_suffix(".jpg"))
        for i, row in dataset.images.iterrows():
            input_img_path = (dataset.images_root / row["relative_path"]).resolve()
            output_img_path = output_img_paths.loc[i]  # pyright: ignore
            if overwrite_images or not output_img_path.is_file():
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                if not to_jpg or row["type"].lower() in [".jpg", ".jpeg"]:
                    shutil.copy(input_img_path, output_img_path)
                else:
                    image = imread(input_img_path)
                    imwrite(output_img_path, image[..., :3])

    if copy_images and to_jpg:
        image_paths_str = dataset.images["relative_path"].apply(
            lambda x: str(x.with_suffix(".jpg"))
        )
    else:
        image_paths_str = dataset.images["relative_path"].apply(str)
    converted_images = dataset.images.drop(["relative_path", "type"], axis=1)
    converted_images["file_name"] = image_paths_str
    converted_bbox = export_bbox(
        dataset.annotations, dataset.images, output_format=box_format
    )
    converted_bboxes = pd.Series(
        converted_bbox[column_names_from_format_string(box_format)].to_numpy().tolist(),
        index=dataset.annotations.index,
        name="bbox",
    )
    converted_annotations = dataset.annotations.drop(
        [*BBOX_COLUMN_NAMES, "category_str"], axis=1
    )
    if "confidence" in converted_annotations:
        converted_annotations = converted_annotations.rename(
            columns={"confidence": "score"}
        )
    converted_annotations = pd.concat([converted_annotations, converted_bboxes], axis=1)
    converted_annotations["iscrowd"] = 0

    if "split" not in converted_images:
        converted_images["split"] = None
    if "split" not in converted_annotations:
        converted_annotations["split"] = dataset.images.loc[
            converted_annotations["image_id"], "split"
        ]

    for split_name, split_imgs in converted_images.groupby("split", dropna=False):
        if add_split_suffix and not pd.isna(split_name):
            output_json_path = output_path / f"{output_file_name}_{split_name}.json"
        else:
            output_json_path = output_path / f"{output_file_name}.json"
        if not overwrite_labels and output_json_path.is_file():
            continue
        print(f"saving {split_name} to {output_json_path.name}...")
        info_dict = {
            "description": "Made with XXII dataset helper",
            "url": "https://xxii.fr",
            "version": version,
            "year": now.year,
            "contributor": contributor,
        }
        images_list = split_imgs.reset_index().to_dict("records")
        if pd.isna(split_name):
            split_annotations = converted_annotations[
                converted_annotations["split"].isna()
            ]
        else:
            split_annotations = converted_annotations[
                converted_annotations["split"] == split_name
            ]
        print(split_annotations)
        print(split_annotations.keys())
        annotations_list = split_annotations.reset_index().to_dict("records")

        categories_list = [
            {
                "supercategory": "",
                "id": cat_id,
                "name": cat_name,
            }  # Not implemented yet,
            for cat_id, cat_name in dataset.label_map.items()
        ]
        coco_dict = {
            "info": info_dict,
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories_list,
        }
        with open(output_json_path, "w") as f:
            json.dump(coco_dict, f, indent=2)
