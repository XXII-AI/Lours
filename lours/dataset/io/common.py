from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import imagesize
import pandas as pd

from lours.dataset import Dataset

from ...utils.bbox_converter import column_names_from_format_string, import_bbox

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


def construct_label_map(annotations: pd.DataFrame) -> dict[int, str]:
    """Construct label map from annotation DataFrame, with ``category_id`` and
    ``category_str`` columns. Get all category string associated with each category id.
    Normally, there should be only one per id

    Args:
        annotations: DataFrame containing category id and category name information.
            Should contain at least ``category_id`` and  ``category_str`` columns.

    Raises:
        ValueError: Inconsistency in category ids and names. The ``id -> name`` mapping
            should be bijective.

    Returns:
        dictionary containing label map, with category id as key, and category name
        as value
    """
    label_map_df = (
        annotations[["category_id", "category_str"]]
        .groupby("category_id")["category_str"]
        .unique()
    )
    label_map_check = label_map_df.apply(len) != 1
    if label_map_check.any():
        print("Problem with label map, some category ids have multiple different names")
        print(label_map_df[label_map_check])
        raise ValueError("Invalid label map")
    label_map = {k: v[0] for k, v in label_map_df.items()}
    # Raise an error if two ids have the same category name
    if len(set(label_map.values())) != len(label_map):
        print("Problem with label map, some category names are present in multiple ids")
        print(label_map)
        raise ValueError("Invalid label map")
    return label_map  # pyright: ignore


def convert_str(string: str) -> str | int | float:
    """String converter tool to read a file, parse and automatically convert the string
    to integer or float if possible. Will first try to convert to int, then float,
    then will return as is.

    Args:
        string: string containing information to be parsed

    Returns:
       converted string, in the most convenient format
    """
    try:
        result = float(string)
        if result == int(result):
            return int(result)
        else:
            return result
    except ValueError:
        return string


def get_relative_image_path(dataset_path: Path, image_path: Path | str) -> Path:
    """Tool function to get relative path between dataset_path and image_path, which
    might be absolute. Used to populate the ``relative_path`` in the images dataframe
    of the dataset or evaluator object, which should check the fact that
    ``dataset.images_root / relative_path`` should always lead to a valid image file

    Args:
        dataset_path: root path of considered dataset
        image_path: image path of a particular image. May be absolute, and need
            to be converted to be relative the dataset_path

    Raises:
        ValueError: image_path is not included in dataset_path. Probably means the
            dataset path is too specific and should be higher in file hierarchy

    Returns:
        Converted image path to be relative to dataset path
    """
    image_path = Path(image_path)
    if image_path.is_absolute():
        try:
            return image_path.relative_to(dataset_path)
        except ValueError as e:
            raise ValueError(
                "Image paths are not contained in given dataset folder. "
                "If you want to use absolute path, "
                "try giving '/' to dataset_path argument"
            ) from e
    else:
        return image_path


def get_images_from_folder(
    folder_path: Path, img_formats: Iterable[str] = IMG_FORMATS
) -> list[Path]:
    """Function to scrape all images in a folder, starting from a list of img formats

    Args:
        folder_path: where to search images
        img_formats: list of file extensions to consider during the globbing

    Returns:
        list of all paths leading to an image with the desired extensions.
    """
    output = []
    for img_format in img_formats:
        output.extend(folder_path.glob(f"**/*.{img_format}"))
    return [path.relative_to(folder_path) for path in output]


def get_image_info(
    image_number: int,
    relative_path: Path,
    absolute_path: Path | None,
    image_info: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Get image information, either from image info dataframe or from image itself,
    getting the image dimension by reading its header

    Args:
        image_number: number of image in the file list. If image_info is not
            available, will be used for image id
        relative_path: path that will be used to find the image in the image_info
            dataframe, if given
        absolute_path: absolute to load the image data directly from the file. Can be
            None if image_info has an entry with the same ``relative_path`` value
        image_info: DataFrame including image size and image id to match the ids of
            another dataset for example. Must have at least ``relative_path``,
            ``width`` and ``height`` columns. Defaults to None.

    Returns:
        dictionary with width height and id keys
    """
    current_image = {"relative_path": relative_path, "type": relative_path.suffix}
    if image_info is None or relative_path not in image_info["relative_path"].values:
        if absolute_path is None:
            raise ValueError(
                "You must at least provide valid image_info or absolute_path"
            )
        width, height = imagesize.get(absolute_path)
        current_image["width"] = width
        current_image["height"] = height
        image_id = image_number
        current_image["id"] = image_id
    else:
        current_image_info = image_info[
            image_info["relative_path"] == relative_path
        ].iloc[0]
        current_image["width"] = current_image_info["width"]
        current_image["height"] = current_image_info["height"]
        image_id = current_image_info.name
        current_image["id"] = image_id
    return current_image


def to_dataset_object(
    images_root: Path,
    label_map: dict[int, str] | None,
    images: Sequence[dict],
    annotations: Sequence[dict],
    box_format: str = "cxwcyh",
    ids_map: dict[int, dict[str, Any]] | None = None,
) -> Dataset:
    """Create the dataset object from aggregated lists of dictionaries

    Args:
        images_root: path where the images are located and from where relative paths
            are given
        label_map: dictionary of category id vs category name
        images: list of image dictionaries. Each dictionary is one image
        annotations: list of annotations dictionaries
        box_format: expected type of box format. See :mod:`lours.utils.bbox_converter`
            Defaults to "cxwcyh"
        ids_map: dictionary to remap classes back to their original id values.
            This is a special case of darknet where the ids are almost always changed
            because they need to be sequential

    Returns:
        created dataset objects with the right category ids
    """
    if images:
        images_df = pd.DataFrame(images).set_index("id")
    else:
        assert (
            len(annotations) == 0
        ), "Got empty image sequence, but a non empty annotation sequence"
        images_df = None
    if annotations:
        annotations_df = pd.DataFrame(annotations)
        if "id" in annotations_df.columns:
            annotations_df = annotations_df.set_index("id")
        else:
            annotations_df.index.name = "id"
    else:
        annotations_df = None
    if images_df is not None and annotations_df is not None:
        bboxes = import_bbox(annotations_df, images_df, input_format=box_format)
        columns_to_drop = column_names_from_format_string(box_format)
        annotations_df = annotations_df.drop(
            columns_to_drop,
            axis=1,
        )
        annotations_df = pd.concat([annotations_df, bboxes], axis=1)
        if label_map is None:
            label_map = construct_label_map(annotations_df)
    dataset = Dataset(
        images_root,
        images_df,
        annotations_df,
        label_map,
    )
    if ids_map is not None:
        id_remapping = {int(i): line["id"] for i, line in ids_map.items()}
        new_label_map = {line["id"]: line["name"] for line in ids_map.values()}
        dataset = dataset.remap_classes(id_remapping, new_label_map)
    return dataset


def parse_annotation_name(
    annotations_file_path: str | Path,
    split_name_mapping: dict[str, list[str]] | None = None,
) -> tuple[str | None, str | None]:
    """Deduce name of dataset and name of split by assuming it is in the form
    '<dataset_name>_<split_name>.<extension>'

    For example, 'coco_train.json' will be parsed to return 'coco' and 'train'

    Args:
        annotations_file_path: name of the annotation file without extension or path to
            the annotation file which name will be parsed.
        split_name_mapping: Dictionary with split names you want to appear in the lours
            dataset as keys and a list of possible words you want this name to replace
            as values. For example, remap split names abbreviations to their full name
            so that "val" becomes "validation". If set to None, will simply map
            variations of 'train', 'valid', 'eval' to them, i.e. 'training' gets
            replaced by 'train', 'val' and 'validation' get replaced by 'valid' and
            'evaluation' and 'test' get replaced by 'eval'. Defaults to None.

    Returns:
        tuple containing two names : dataset name and split name. They can be none in
        the case parsing was not successful.

    Example:
        >>> parse_annotation_name("my_dataset_test")
        ('my_dataset', 'eval')
        >>> parse_annotation_name("my_dataset_hello", {"hey": ["hello", "hi"]})
        ('my_dataset', 'hey')
        >>> parse_annotation_name("my_dataset")
        ('my', 'dataset')
        >>> parse_annotation_name("mydataset")
        ('mydataset', None)
    """
    if split_name_mapping is None:
        split_name_mapping = {
            "train": ["train", "training"],
            "valid": ["val", "valid", "validation"],
            "eval": ["eval", "evaluation", "test"],
        }
    if isinstance(annotations_file_path, Path):
        name = annotations_file_path.stem
    else:
        name = annotations_file_path
    if "_" in name:
        parsed_name, parsed_split = name.rsplit("_", maxsplit=1)
        for possible_split, possible_names in split_name_mapping.items():
            if parsed_split in possible_names:
                parsed_split = possible_split
        return parsed_name, parsed_split
    return name, None
