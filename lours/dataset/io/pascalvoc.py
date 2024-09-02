from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

import pandas as pd
import xmltodict
from tqdm import tqdm

from lours.dataset import Dataset
from lours.utils.bbox_converter import import_bbox


def from_pascalVOC_generic(
    annotations_root: Path,
    images_root: Path | str | None = None,
    split_folder: Path | str | None = None,
    split_values: Iterable[str] | str = ("train", "val"),
) -> Dataset:
    """Load a dataset in pascalVOC format

    See `specifications`__ (only Object detection)

        .. __: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html

    See Also:
        :func:`.from_pascalVOC_detection`

    Notes:
        - This has been tested against PascalVOC2012
        - If loading official detection splits, not all images will be assigned a split
          value.
        - For objects with "parts" (like hands for persons), a new object will be
          created and a ``body_id`` column will link to the corresponding root object.
        - ``actions.<value>`` columns are converted to boolean and included in the
          booleanized column ``actions``.
        - ``difficult``, ``truncated`` and ``occluded`` columns are converted to
          boolean.

    Args:
        annotations_root: Folder containing the xml files containing annotations
        images_root: Folder containing the image files. Path to images are given in the
            corresponding annotation files, relative to this folder. If set to None,
            will be assumed to be the same as ``annotations_root``. Defaults to None.
        split_folder: Folder containing txt file for each split. Files are named
            ``<split>.txt`` and contain all the file name without extension contained
            in that split. If set to None, will be assumed to be the same as
            ``annotations_root``. Defaults to None.
        split_values: split value or list of split values to read. Following the
            aforementioned syntax, will try to open the corresponding split files and
            assign this split value to the corresponding images. Note that split values
            need to be exclusive to each other. For example, you cannot load both
            "train" and "trainval" splits. Defaults to ("train", "val").

    Returns:
        Loaded dataset with split values assigned
    """
    if images_root is None:
        images_root = annotations_root
    if split_folder is None:
        split_folder = annotations_root
    annotations_root = Path(annotations_root)
    images_root = Path(images_root)
    split_folder = Path(split_folder)
    if isinstance(split_values, str):
        split_values = [split_values]
    xml_files = list(annotations_root.glob("**/*.xml"))

    def image_set(split: str) -> set[str]:
        try:
            with open(split_folder / f"{split}.txt") as f:
                split_images = f.read().strip().split("\n")
            return set(split_images)
        except FileNotFoundError:
            return set()

    split_files: dict[str, set[str]] = {}
    for value in split_values:
        split_files[value] = image_set(value)

    for s1, s2 in combinations(split_files.keys(), 2):
        overlap = split_files[s1].intersection(split_files[s2])
        assert not overlap, f"Splits {s1} and {s2} have non null overlap : {overlap}"

    reversed_splits = {}
    for split_name, split_set in split_files.items():
        for file_name in split_set:
            reversed_splits[file_name] = split_name
    annotations_dicts = []

    image_data_list = {}

    current_object_id = 0

    for image_id, xml_file in enumerate(tqdm(xml_files)):
        annotation_dict = xmltodict.parse(xml_file.read_text())["annotation"]

        image_data = {}
        image_data["relative_path"] = annotation_dict["filename"]
        image_data["width"] = int(annotation_dict["size"]["width"])
        image_data["height"] = int(annotation_dict["size"]["height"])
        image_data["split"] = reversed_splits.get(xml_file.stem, None)

        image_data_list[image_id] = image_data

        objects = annotation_dict["object"]
        if isinstance(objects, dict):
            objects = [objects]
        object_parts = []
        for object in objects:
            object["id"] = current_object_id
            object["image_id"] = image_id
            if "part" in object:
                parts = object.pop("part")
                if isinstance(parts, dict):
                    parts = [parts]
                for p in parts:
                    p["body_id"] = current_object_id
                object_parts.extend(parts)
            if "point" in object:
                point = object.pop("point")
                x, y = point["x"], point["y"]
                object_parts.append(
                    {
                        "bndbox": {"xmax": x, "xmin": x, "ymax": y, "ymin": y},
                        "image_id": image_id,
                        "body_id": current_object_id,
                        "name": "person of interest",
                    }
                )
            current_object_id += 1
        for part in object_parts:
            part["id"] = current_object_id
            part["image_id"] = image_id
            current_object_id += 1
        annotations_dicts.extend(objects)
        annotations_dicts.extend(object_parts)
    images_df = pd.DataFrame.from_dict(image_data_list, orient="index")
    images_df.index = images_df.index.rename("id")
    annotations_df = pd.json_normalize(annotations_dicts).set_index("id")

    annotations_df = annotations_df.astype(
        {"body_id": pd.Int64Dtype()}, errors="ignore"
    )
    action_columns = [
        name for name in annotations_df.columns if name.startswith("actions.")
    ]
    annotations_df[action_columns] = (
        annotations_df[action_columns]
        .replace({"0": False, "1": True})
        .fillna(False)
        .astype(bool)
    )

    to_boolean = list(
        {"difficult", "occluded", "truncated"}.intersection(set(annotations_df.columns))
    )
    annotations_df[to_boolean] = (
        annotations_df[to_boolean]
        .replace(
            {
                "0": False,
                "1": True,
            }
        )
        .astype(pd.BooleanDtype)
    )

    bounding_boxes = import_bbox(
        annotations_df[["bndbox.xmin", "bndbox.xmax", "bndbox.ymin", "bndbox.ymax"]]
        .astype(float)
        .to_numpy(),
        images_df,
        image_ids=annotations_df["image_id"],
        input_format="XXYY",
    )

    annotations_df = pd.concat(
        [
            annotations_df.drop(
                [
                    "bndbox.xmin",
                    "bndbox.xmax",
                    "bndbox.ymin",
                    "bndbox.ymax",
                ],
                axis=1,
            ),
            bounding_boxes,
        ],
        axis=1,
    )

    label_map = dict(enumerate(annotations_df["name"].unique()))
    reverse_label_map = {v: k for k, v in label_map.items()}
    annotations_df["category_id"] = annotations_df["name"].replace(reverse_label_map)
    annotations_df = annotations_df.rename(columns={"name": "category_str"})

    dataset = Dataset(
        images_root=images_root,
        images=images_df,
        annotations=annotations_df,
        label_map=label_map,
    )

    dataset.booleanized_columns["annotations"] = {"actions"}

    return dataset


def from_pascalVOC_detection(input_folder: Path) -> Dataset:
    """Load a pascalVOC detection dataset that follows the official structure.

    Folder is assumed to contain three sub-folders:

    - "Annotations" containing the annotation xml files
    - "JPEGImages" containing the images files
    - "ImageSets/Main" containing the detection split files

    See `specifications`__

        .. __: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html

    See Also:
        :func:`.from_pascalVOC_generic`

    Notes:
        - This has been tested against PascalVOC2012
        - If loading official detection splits, not all images will be assigned a split
          value.
        - For objects with "parts" (like hands for persons), a new object will be
          created and a ``body_id`` column will link to the corresponding root object.
        - ``actions.<value>`` columns are converted to boolean and included in the
          booleanized column ``actions``.
        - ``difficult``, ``truncated`` and ``occluded`` columns are converted to
          boolean.
        - The dataset will remove images without split. If you wish to load all images
          with available annotation, use :func:`.from_pascalVOC_generic`.

    Args:
        input_folder: Folder containing annotations, images, and split folders.

    Returns:
        Loaded dataset
    """
    annotations_folder = input_folder / "Annotations"
    images_root = input_folder / "JPEGImages"
    split_folder = input_folder / "ImageSets" / "Main"
    pascal_dataset = from_pascalVOC_generic(
        annotations_folder, images_root, split_folder
    )
    # Remove images that neither in the train nor in the valid split
    return pascal_dataset.loc[~pascal_dataset.images["split"].isna()]
