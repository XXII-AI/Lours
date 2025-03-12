import json
import shutil
from collections.abc import Iterable
from os.path import relpath
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from imageio.v3 import imread, imwrite
from jsonschema_rs import validator_for
from tqdm.auto import tqdm

from lours.dataset import Dataset
from lours.utils import BBOX_COLUMN_NAMES
from lours.utils.testing import assert_columns_properly_normalized, assert_images_valid

from .common import construct_label_map
from .schema_util import (
    fill_with_dtypes_and_default_value,
    flatten_schema,
    get_enums,
    get_remapping_dict_from_names,
    get_remapping_dict_from_schema,
    load_json_schema,
    remap_dict,
)


def load_caipy_annot_folder(
    folder_path: Path, schema: dict | None = None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Glob all json in folder path and construct image and annotation dataframe

    Args:
        folder_path: folder where we will search for json files
        schema: Optional JSON schema dict used to check the conformity of loaded
            JSON files.

    Returns:
        A pair of dataframes, representing image and annotations data,
        most likely used to construct the dataset object.
    """
    annotations = []
    images = []
    annotations_files = list(folder_path.glob("**/*.json"))
    if schema is not None:
        validator = validator_for(schema)
    else:
        validator = None
    for annot_file in tqdm(annotations_files):
        with open(annot_file) as f:
            frame_data = json.load(f)
        if validator is not None:
            validator.validate(frame_data)
        if "type" in frame_data.keys():
            assert (
                frame_data["type"] == "instances"
            ), "Only instance type supported for now"
        image_data = frame_data["image"]
        image_file_name = Path(image_data.pop("file_name"))
        assert annot_file.stem == image_file_name.stem, (
            "bad formatting, annotation file and image file_name should only differ by"
            f" the extension, got {annot_file.name} and"
            f" {image_file_name} instead"
        )
        image_path = annot_file.parent.relative_to(folder_path) / image_file_name
        image_data["relative_path"] = image_path
        image_data["type"] = image_path.suffix
        images.append(image_data)
        for annot in frame_data["annotations"]:
            annot["image_id"] = image_data["id"]
        annotations.extend(frame_data["annotations"])
    if images:
        # Sometimes, one key is either a sub dictionary or nan.
        # As a consequence, the column "key" will be full of nans while columns
        # "key.subkey" exist. remove columns full of NaN to avoid this situation.
        images = pd.json_normalize(images).set_index("id").dropna(axis=1, how="all")
        set_to_none_columns = images.select_dtypes("object").columns
        images[set_to_none_columns] = images[set_to_none_columns].replace(np.nan, None)
        assert_columns_properly_normalized(images)
    else:
        images = None
    if annotations:
        annotations = (
            pd.json_normalize(annotations).set_index("id").dropna(axis=1, how="all")
        )
        set_to_none_columns = annotations.select_dtypes("object").columns
        annotations[set_to_none_columns] = annotations[set_to_none_columns].replace(
            np.nan, None
        )
        # BBoxes are currently lists of 4 elements in a "bbox" column.
        # Convert each list element in the corresponding column for the dataset object
        bboxes = pd.DataFrame(list(annotations["bbox"]), index=annotations.index)
        annotations = annotations.drop("bbox", axis=1)
        for i, name in enumerate(BBOX_COLUMN_NAMES):
            annotations[name] = bboxes[i]
        assert_columns_properly_normalized(annotations)
    else:
        annotations = None

    return images, annotations


def load_caipy_split(
    images_folder: Path,
    annotations_folder: Path,
    dataset_name: str | None = None,
    split_name: str | None = None,
    schema: dict | None = None,
) -> Dataset:
    """Load a particular caipy split folder and convert it to a lours Dataset

    Args:
        images_folder: folder where images are stored
        annotations_folder: folder where annotations are stored as json files
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone. If not specified,
            the dataset name will be the name of the root folder.
        split_name: name of the split to give to ``split`` column of images DataFrame.
            Defaults to ``None``.
        schema: JSON schema dict used to check the conformity of loaded JSON files.
            If set to ``None``, will not check the conformity. Defaults to ``None``.

    Raises:
        ValueError: If image ids are not mutually exclusives

    Returns:
        Dataset containing only one split from caipy, expected to be merged with other
        caipy splits
    """
    images, annotations = load_caipy_annot_folder(annotations_folder, schema)
    if images is not None:
        if not images.index.is_unique:
            raise ValueError("two images have the same id")
        if split_name is not None:
            images["split"] = split_name

    label_map = (
        construct_label_map(annotations=annotations)
        if annotations is not None
        else None
    )
    return Dataset(
        images_root=images_folder,
        images=images,
        annotations=annotations,
        label_map=label_map,
        dataset_name=dataset_name,
    )


def from_caipy(
    dataset_path: Path | str,
    dataset_name: str | None = None,
    split: str | None = None,
    splits_to_read: str | Iterable[str] | None = None,
    use_schema: bool = False,
    json_schema: dict | str | Path | None = "default",
    booleanize: bool = True,
) -> Dataset:
    """Load a dataset stored in the cAIpy format

    See `specifications`__

    .. __: UPDATE-ME

    This will error if

    - two annotations have the same ``category_id`` but not the same ``category_str``
    - two annotations have a different ``category_id``  but the same ``category_str``
    - two images have the same ``file_name``, but not the same ``id``

    Args:
        dataset_path: folder root of dataset. Should contain the folders
            "Images" and "Annotations".
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone. If not specified,
            the dataset name will be the name of the root folder.
        split: if data is at the root of Images and Annotations folder, the split
            will be set to this option. Defaults to ``None``
        splits_to_read: if given, will only read the specified splits. Useful for a
            faster loading.
        use_schema: If set to True, and ``json_schema`` is not None, will use schema for
            validation and formatting (see option ``json_schema``)
        json_schema: schema dictionary or Path to a schema that json files will be
            tested against for compliance. If its not a dictionary, it can be either a
            url, or a local path. If set to None, or
            ``use_schema`` is set to False, will not perform any test.
            Defaults to default schema.
        booleanize: In the case some attributes are array of enum with unique
            elements, they will be booleanized
            (see :func:`~lours.dataset.Dataset.booleanize`).
            Note that this option is only used if `json_schema`` is not None and
            ``use_schema`` is set to True. Defaults to True.

    Raises:
        ValueError: Inconsistency between two annotations or images
            (see description above)


    Returns:
        Loaded dataset object


    See Also:
        - :func:`from_caipy_generic`
        - :ref:`Tutorial on schemas </notebooks/6_demo_schemas.ipynb>`
        - :ref:`Tutorial on booleanization </notebooks/7_demo_booleanize.ipynb>`
        - `cAIpy specifications <UPDATE-ME>`_
    """  # noqa: E501
    dataset_path = Path(dataset_path)
    annotations_folder = dataset_path / "Annotations"
    images_folder = dataset_path / "Images"
    dataset_name = dataset_name if dataset_name is not None else dataset_path.name
    return from_caipy_generic(
        images_folder,
        annotations_folder,
        dataset_name,
        split,
        splits_to_read,
        use_schema,
        json_schema,
        booleanize,
    )


def from_caipy_generic(
    images_folder: Path | str | None,
    annotations_folder: Path | str,
    dataset_name: str | None = None,
    split: str | None = None,
    splits_to_read: str | Iterable[str] | None = None,
    use_schema: bool = False,
    json_schema: dict | str | Path | None = "default",
    booleanize: bool = True,
) -> Dataset:
    """Load a dataset stored in the cAIpy format, but you can specify images and
    annotations folders rather than giving a folder with Images and Annotations
    sub-folders. This gives much more flexibility, especially when working predictions
    and annotations variations.

    See `specifications`__

    .. __: UPDATE-ME

    this will error if

    - two annotations have the same ``category_id`` but not the same ``category_str``
    - two annotations have a different ``category_id``  but the same ``category_str``
    - two images have the same ``file_name``, but not the same ``id``

    Args:
        images_folder: folder root of images.
        annotations_folder: folder root of annotations.
        dataset_name: If specified, will be the dataset name, used when showing the
            dataset or exporting in other formats such as fiftyone.
        split: if data is at the root of Images and Annotations folder, the split
            will be set to this option. Defaults to ``None``
        splits_to_read: if given, will only read the specified splits. Useful for a
            faster loading.
        use_schema: If set to True, and ``json_schema`` is not None, will use schema for
            validation and formatting (see option ``json_schema``)
        json_schema: schema dictionary or Path to a schema that json files will be
            tested against for compliance. If its not a dictionary, it can be either a
            url or a local path. If set to None, or
            ``use_schema`` is set to False, will not perform any test.
            Defaults to default schema.
        booleanize: In the case some attributes are array of enum with unique
            elements, they will be booleanized
            (see :func:`~lours.dataset.Dataset.booleanize`).
            Note that this option is only used if `json_schema`` is not None and
            ``use_schema`` is set to True. Defaults to True.

    Raises:
        ValueError: Inconsistency between two annotations or images
            (see description above)


    Returns:
        Loaded dataset object

    See Also:
        - :func:`from_caipy`
        - :ref:`Tutorial on schemas </notebooks/6_demo_schemas.ipynb>`
        - :ref:`Tutorial on booleanization </notebooks/7_demo_booleanize.ipynb>`
        - `cAIpy specifications <UPDATE-ME>`_
    """  # noqa: E501
    if use_schema and json_schema is not None:
        if isinstance(json_schema, dict):
            schema = json_schema
        else:
            schema = load_json_schema(json_schema)
    else:
        schema = None
    annotations_folder = Path(annotations_folder)
    if images_folder is not None:
        images_folder = Path(images_folder)
    else:
        images_folder = annotations_folder.parent / "Images"
        print(f"specifying a fictive path for images : {images_folder}")
    if isinstance(splits_to_read, str):
        splits_to_read = [splits_to_read]
    if splits_to_read is None:
        selected_splits = ["train", "valid", "eval"]
    else:
        selected_splits = splits_to_read

    dataset = Dataset(images_root=images_folder, dataset_name=dataset_name)
    for split_to_read in selected_splits:
        split_folder = annotations_folder / split_to_read
        if not split_folder.is_dir():
            continue
        split_dataset = load_caipy_split(
            images_folder=images_folder / split_to_read,
            annotations_folder=split_folder,
            dataset_name=dataset_name,
            split_name=split_to_read,
            schema=schema,
        )
        if len(split_dataset) == 0:
            continue
        dataset += split_dataset

    if len(dataset) == 0 and splits_to_read is None:
        dataset = load_caipy_split(
            images_folder=images_folder,
            annotations_folder=annotations_folder,
            dataset_name=dataset_name,
            split_name=split,
            schema=schema,
        )

    if schema is not None:
        image_schema = schema["properties"]["image"]
        annotation_schema = schema["properties"]["annotations"]["items"]
        if booleanize:
            image_enums = get_enums(image_schema)
            annotation_enums = get_enums(annotation_schema)
            dataset = dataset.booleanize(
                missing_ok=True, **(annotation_enums | image_enums)
            )
        dataset.images = fill_with_dtypes_and_default_value(
            image_schema, dataset.images
        )
        dataset.annotations = fill_with_dtypes_and_default_value(
            annotation_schema, dataset.annotations
        )
    return dataset


def split_to_caipy(
    dataset: Dataset,
    split_images_folder: Path | None,
    split_annotations_folder: Path,
    schema: dict | None = None,
    copy_images: bool = True,
    to_jpg: bool = True,
    overwrite_images: bool = True,
    overwrite_labels: bool = True,
    flatten_paths: bool = True,
) -> None:
    """Save a particular split to cAIpy. images and annotations folder must be given,
    as it can be the root of "Images" and "Annotations",
    or a subfolder based on split name, e.g. "Images/train"

    Note:
        Unless specified otherwise, relative paths of images a flattened during the
        export, which modifies the dataset if the images and annotations
        were stored in subfolders, but will put all images and annotations in their
        respective root folder.

    Note:
        If schema is not given, the nested dictionary will be deduced from column names
        with the separator "."

    Args:
        dataset: dataset object to save. Normally, should be a unique split
        split_images_folder: dataset where to save images, either as links or files.
            If None, will not save images. This is useful when you just want to save
            predictions or a variation of annotations.
        split_annotations_folder: dataset where to save caipyjson files.
        schema: JSON schema dict used to check the conformity of output JSON files.
            It will also be used to remove columns for fields no included
            in the schema. If set to ``None``, will not check the conformity.
            Defaults to ``None``.
        copy_images: If set to False, will create a symbolic link instead of copying.
            Much faster, but needs to keep original images in the same relative path.
            Defaults to False.
        to_jpg: if True, will convert images to jpg if needed. Defaults to True.
        overwrite_images: if set to False, will skip images that are already copied.
            Defaults to True.
        overwrite_labels: if set to False, will skip annotation that are already
            created. Defaults to True.
        flatten_paths: if set to True, will put all files in the root Annotations and
            Images folders by replacing folder separation ("/") with "_" in relative
            path. Defaults to True
    """
    if schema is not None:
        validator = validator_for(schema)
    else:
        validator = None
    # Get back to the list of 4 elements format
    converted_bboxes = pd.Series(
        dataset.annotations[BBOX_COLUMN_NAMES].to_numpy().tolist(),
        index=dataset.annotations.index,
        name="bbox",
    )
    # Remove useless columns and append the new one
    converted_annotations = dataset.annotations.drop(BBOX_COLUMN_NAMES, axis=1)
    converted_annotations = pd.concat([converted_annotations, converted_bboxes], axis=1)
    converted_annotations = converted_annotations.reset_index()
    # Get back the relative_path to a simple string
    n_images = len(dataset.images)
    if schema is None:
        image_remapping_dict = get_remapping_dict_from_names(
            frozenset(
                [*dataset.get_image_attributes(), "file_name", "id", "width", "height"]
            )
        )
        annotations_remapping_dict = get_remapping_dict_from_names(
            frozenset(
                [
                    *dataset.get_annotations_attributes(),
                    "id",
                    "bbox",
                    "category_id",
                    "category_str",
                ]
            )
        )
    else:
        image_schema = schema["properties"]["image"]
        annotations_schema = schema["properties"]["annotations"]["items"]
        image_remapping_dict = get_remapping_dict_from_schema(image_schema)
        saved_image_keys = flatten_schema(image_schema)
        lost_image_columns = set(dataset.get_image_attributes()) - set(saved_image_keys)
        if lost_image_columns:
            warn(
                "These column in self.images will be lost because they don't follow"
                f" the specified json schema: {', '.join(lost_image_columns)}",
                RuntimeWarning,
            )
        annotations_remapping_dict = get_remapping_dict_from_schema(annotations_schema)
        saved_annot_keys = flatten_schema(annotations_schema)
        lost_annot_columns = set(dataset.get_annotations_attributes()) - set(
            saved_annot_keys
        )
        if lost_annot_columns:
            warn(
                "These column in self.annotations will be lost because they don't"
                " follow the specified json schema:"
                f" {', '.join(lost_annot_columns)}",
                RuntimeWarning,
            )

    for image_id, image_data in tqdm(dataset.images.iterrows(), total=n_images):
        assert isinstance(image_id, int)
        instances = converted_annotations[converted_annotations["image_id"] == image_id]
        input_image_path = (dataset.images_root / image_data["relative_path"]).resolve()
        # Handle the case of images coming from a cAIpy, which already have the
        # structure {split}/filename in their relative path, which we don't want
        output_relative_path = image_data["relative_path"]
        if output_relative_path.parts[0] in ["train", "valid", "eval"]:
            output_relative_path = Path(*output_relative_path.parts[1:])
        if flatten_paths:
            output_relative_path = Path("_".join(output_relative_path.parts))
        output_filename = output_relative_path.name

        if split_images_folder is not None:
            output_image_path = (split_images_folder / output_relative_path).resolve()
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            if to_jpg:
                output_image_path = output_image_path.with_suffix(".jpg")
                output_filename = output_image_path.name
            # Copy image to destination. Do nothing if the image already exists,
            if overwrite_images or not output_image_path.is_file():
                if to_jpg and image_data["type"].lower() not in [".jpg", ".jpeg"]:
                    image = imread(input_image_path)
                    imwrite(output_image_path, image[..., :3])
                elif copy_images:
                    shutil.copy(input_image_path, output_image_path)
                else:
                    output_image_path.unlink(missing_ok=True)
                    output_image_path.symlink_to(
                        relpath(input_image_path, output_image_path.parent)
                    )

        output_json_path = split_annotations_folder / output_relative_path.with_suffix(
            ".json"
        )
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_json_path.is_file() or overwrite_labels:
            # Get only annotations associated with this image
            image_dict = {
                "file_name": output_filename,
                "id": int(image_id),
                "width": int(image_data["width"]),
                "height": int(image_data["height"]),
                **image_data[dataset.get_image_attributes()],
            }
            image_dict = remap_dict(image_dict, image_remapping_dict)
            annot_list = instances.to_dict("records")
            annot_list = list(
                map(lambda x: remap_dict(x, annotations_remapping_dict), annot_list)
            )
            annotations_dict = {
                "image": image_dict,
                "type": "instances",
                "annotations": annot_list,
            }
            if validator is not None:
                validator.validate(annotations_dict)

            with open(output_json_path, "w") as f:
                json.dump(annotations_dict, f, indent=2)


def dataset_to_caipy(
    dataset: Dataset,
    output_path: Path | str,
    use_schema: bool = False,
    json_schema: str | Path | None = "default",
    copy_images: bool = True,
    to_jpg: bool = True,
    overwrite_images: bool = True,
    overwrite_labels: bool = True,
    flatten_paths: bool = True,
) -> None:
    """Save dataset to cAIpy format
    Note that depending on the splits present in your dataset,
    the folder structure might change

    Note:
        Unless specified otherwise, relative paths of images a flattened during the
        export, which modifies the dataset if the images and annotations
        were stored in subfolders, but will put all images and annotations of a
        particular split in their respective root folder.

    Note:
        If schema is not given, the nested dictionary will be deduced from column names
        with the separator "."

    Args:
        dataset: dataset to save
        output_path: root folder where the dataset folder structure will be created.
        use_schema: If set to True, and ``json_schema`` is not None, will use schema for
            validation and formatting (see option ``json_schema``)
        json_schema: Path to a schema that output json dicts will be tested against for
            compliance. They will also be used to remove columns for fields no included
            in the schema. Can be either a url or a local path.
            If set to None, or ``use_schema`` is set to False,
            will not perform any test or reformatting. Defaults to default schema.
        copy_images: If set to False, will create a symbolic link instead of copying.
            Much faster, but needs to keep original images in the same relative path.
            Defaults to False.
        to_jpg: if True, will convert images to jpg if needed. Defaults to True.
        overwrite_images: if set to False, will skip images that are already copied.
            Defaults to True.
        overwrite_labels: if set to False, will skip annotation that are already
            created. Defaults to True.
        flatten_paths: if set to True, will put all files in the root Annotations and
            Images folders by replacing folder separation ("/") with "_" in relative
            path. Defaults to True

    """
    output_path = Path(output_path)
    images_folder = output_path / "Images"
    annotations_folder = output_path / "Annotations"
    return dataset_to_caipy_generic(
        dataset,
        images_folder,
        annotations_folder,
        use_schema,
        json_schema,
        copy_images,
        to_jpg,
        overwrite_images,
        overwrite_labels,
        flatten_paths,
    )


def dataset_to_caipy_generic(
    dataset: Dataset,
    output_images_folder: Path | str | None,
    output_annotations_folder: Path | str,
    use_schema: bool = False,
    json_schema: str | Path | None = "default",
    copy_images: bool = True,
    to_jpg: bool = True,
    overwrite_images: bool = True,
    overwrite_labels: bool = True,
    flatten_paths: bool = True,
) -> None:
    """Save dataset to cAIpy format
    Note that depending on the splits present in your dataset,
    the folder structure might change

    Notes:
        - Unless specified otherwise, relative paths of images a flattened during the
          export, which modifies the dataset if the images and annotations
          were stored in subfolders, but will put all images and annotations of a
          particular split in their respective root folder.
        - If schema is not given, the nested dictionary will be deduced from column names
          with the separator "."

    Args:
        dataset: dataset to save
        output_images_folder: root folder where the images will be saved. If None, will
            not save images. Useful when only saving predictions or a variations of
            annotations.
        output_annotations_folder: root folder where the json file will be saved.
        use_schema: If set to True, and ``json_schema`` is not None, will use schema for
            validation and formatting (see option ``json_schema``)
        json_schema: Path to a schema that output json dicts will be tested against for
            compliance. They will also be used to remove columns for fields no included
            in the schema. Can be either a url or a localt path.
            If set to None, or ``use_schema`` is set to False,
            will not perform any test. Defaults to the example schema.
        copy_images: If set to False, will create a symbolic link instead of copying.
            Much faster, but needs to keep original images in the same relative path.
            Defaults to False.
        to_jpg: if True, will convert images to jpg if needed. Defaults to True.
        overwrite_images: if set to False, will skip images that are already copied.
            Defaults to True.
        overwrite_labels: if set to False, will skip annotation that are already
            created. Defaults to True.
        flatten_paths: if set to True, will put all files in the root Annotations and
            Images folders by replacing folder separation ("/") with "_" in relative
            path. Defaults to True

    """
    if output_images_folder is not None:
        try:
            assert_images_valid(dataset, load_images=False)
        except AssertionError as e:
            raise ValueError(
                "Dataset images are missing, check that the images root folder is the"
                " right one"
            ) from e
    if use_schema and json_schema is not None:
        schema = load_json_schema(json_schema)
    else:
        schema = None
    output_dataset = dataset.debooleanize()
    if "split" in dataset.images.columns:
        splits = dataset.images["split"].unique().tolist()
    else:
        splits = []
    output_annotations_folder = Path(output_annotations_folder)
    if output_images_folder is not None:
        output_images_folder = Path(output_images_folder)
        output_images_folder.mkdir(exist_ok=True, parents=True)
    output_annotations_folder.mkdir(exist_ok=True, parents=True)
    # If no split, or only split evaluate to false (empty string or null value)
    # save the dataset without split subfolders
    if not splits or (len(splits) == 1 and pd.isna(splits[0])):
        print("Saving cAIpy dataset without split")
        split_to_caipy(
            output_dataset,
            output_images_folder,
            output_annotations_folder,
            schema,
            copy_images,
            to_jpg,
            overwrite_images,
            overwrite_labels,
            flatten_paths,
        )
    else:
        if any(pd.isna(split) for split in splits):
            raise AssertionError(
                "Dataset cannot have both data with known split and with unknown split"
            )
        for split_name in splits:
            print(f"Saving {split_name} split...")
            if output_images_folder is not None:
                split_images_folder = output_images_folder / f"{split_name}"
                split_images_folder.mkdir(exist_ok=True)
            else:
                split_images_folder = None
            split_annotations_folder = output_annotations_folder / f"{split_name}"
            split_annotations_folder.mkdir(exist_ok=True)
            split_to_caipy(
                output_dataset.get_split(split_name),
                split_images_folder,
                split_annotations_folder,
                schema,
                copy_images,
                to_jpg,
                overwrite_images,
                overwrite_labels,
                flatten_paths,
            )
