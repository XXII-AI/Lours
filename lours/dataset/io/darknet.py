import json
import shutil
from collections.abc import Iterable, Mapping
from os.path import relpath
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from imageio.v3 import imread, imwrite
from tqdm.auto import tqdm

from lours.utils.bbox_converter import column_names_from_format_string, export_bbox
from lours.utils.testing import assert_images_valid

from ..dataset import Dataset
from .common import (
    convert_str,
    get_image_info,
    get_images_from_folder,
    get_relative_image_path,
    to_dataset_object,
)

CX_COLUMN, W_COLUMN, CY_COLUMN, H_COLUMN = column_names_from_format_string("cxwcyh")


def bbox_to_txt(bbox_df: pd.DataFrame) -> list[str]:
    """Convert dataframe of yolo bboxes to a list of strings to be written in
    a file readable by darknet

    Args:
        bbox_df: bboxes dataframe. Must be in yolo format (see function above)

    Returns:
        list of string describing bboxes according to darknet format
    """
    lines = []
    for _, box in bbox_df.iterrows():
        box_string = (
            f"{box['category_id']} "
            f"{box[CX_COLUMN]} "
            f"{box[CY_COLUMN]} "
            f"{box[W_COLUMN]} "
            f"{box[H_COLUMN]}"
        )
        lines.append(f"{box_string}\n")
    return lines


def txt_to_bbox(text_lines: Iterable[str], image_id: int) -> list[dict[str, Any]]:
    """Read lines coming from a darknet txt annotation file and convert it
    to a list of annotation dictionaries

    Args:
        text_lines: list of text lines in a particular folder all given lines are
            assumed to come from the same image
        image_id: image index of corresponding image

    Returns:
        list of annotations for the particular image. bboxes are in the darknet format,
        i.e. with relative coordinates
    """
    bboxes = []
    for line in text_lines:
        cat_id, x, y, width, height, *rest = map(convert_str, line.split())
        bbox = {
            "category_id": cat_id,
            CX_COLUMN: x,
            CY_COLUMN: y,
            W_COLUMN: width,
            H_COLUMN: height,
            "image_id": image_id,
        }
        if rest:
            bbox["confidence"] = rest[0]
        bboxes.append(bbox)
    return bboxes


def open_data_file(data_file: Path) -> dict[str, str | int | float]:
    """Open a ``.data`` file typically used in darknet.

    Args:
        data_file: path to ``.data`` file

    Returns:
        dictionary of values contained in the .data file
    """
    data_dict = {}
    with open(data_file) as f:
        lines = f.read().split("\n")
    for line in lines:
        if line:
            try:
                key, value = line.split(" = ")
            except ValueError:
                key = line.strip("=").strip()
                value = ""
            data_dict[key] = convert_str(value)
    return data_dict


def write_data_file(data: dict[str, str | int], data_file: Path) -> None:
    """Save a dictionary to a ``.data`` file typically used in darknet.
    Note that only the typical keys expected by darknet are considered,
    i.e. classes, train, valid, names and backup

    Args:
        data: content to save to the ``.data`` file
        data_file: path where to save the given data dictionary
    """
    keys = ["classes", "train", "valid", "names", "backup"]
    with open(data_file, "w") as f:
        for key in keys:
            f.write(f"{key} = {data.get(key, '')}\n")


def yolov5_img_path_to_label_path(img_path: Path) -> Path:
    """Convert the path of an image to its corresponding label

    That is, if the path has a folder named "images", rename that folder into "labels"
    and change the image extension to "txt"

    See `Yolov5's image2labels <https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py#L424>`__ .

    Args:
        img_path: path to an image file

    Returns:
        corresponding label file as it would have been searched for by yolov5
    """  # noqa: E501
    annotation_path = img_path.with_suffix(".txt")
    folder_parts = list(annotation_path.parts)
    if "images" in folder_parts:
        # In case they are multiple occurrences of images folder, use the last one
        images_folder_index = len(folder_parts) - 1 - folder_parts[::-1].index("images")
        folder_parts[images_folder_index] = "labels"
        annotation_path = Path(*folder_parts)
    return annotation_path


def deduce_data_file_and_dataset_folder(
    dataset_path: Path | str | None = None,
    data_file: Path | str | None = None,
    default_data_file: str = "train_job.data",
) -> tuple[Path, Path]:
    """Get dataset path and data file, deduce one from another if needed."""
    if dataset_path is None and data_file is None:
        raise ValueError("dataset_path and data_file cannot be both None !")
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_dir():
            raise ValueError("Dataset path must be a path to valid directory")
        if data_file is None:
            data_file = dataset_path / default_data_file
            if not data_file.is_file():
                raise ValueError(f"Data file {data_file} does not exist")
    if data_file is not None:
        data_file = Path(data_file)
        if not data_file.is_file():
            raise ValueError(f"Data file {data_file} does not exist")
        if dataset_path is None:
            dataset_path = data_file.parent

    # Do a final assert, otherwise pyright will complain
    assert dataset_path is not None and data_file is not None
    return dataset_path, data_file


def from_darknet(
    dataset_path: Path | str | None = None,
    data_file: Path | str | None = None,
    ids_map: dict[int, dict[str, Any]] | str | Path | None = None,
    image_info: pd.DataFrame | None = None,
) -> Dataset:
    """Creates dataset object from a darknet dataset. Note that category ids and
    image ids are not given in the dataset format and thus can only be sequential
    As such, if we want to convert the dataset back to another format that keeps track
    of image and category ids, we need to give image_info and class mapping from
    an external source. Here we expect it to be contained in a json label map for
    annotations and a DataFrame with similar columns as in the final dataset's images
    DataFrame for images

    Args:
        dataset_path: folder containing the dataset, from which the relative path
            are given. If not set, will use ``data_file``'s parent directory.
            Defaults to None
        data_file: data file containing info about names, lists of train and
            validation images. Can be either a .data file or a .yml file (for yolov5).
            If not set, will use the file ``train_job.data`` at the root of
            ``dataset_path``. Defaults to None.
        ids_map: Optional dictionary containing the id_remapping that was initially
            applied to create the darknet dataset. Will reverse it to get back to the
            original class mapping. The dictionary must have darknet dataset's category
            ids (in sequential order then) as keys and with corresponding values that
            are dictionaries containing ``name`` and ``id`` keys relative to this
            class. Note that this can also be a path to a json file containing the
            dictionary. Defaults to None.
        image_info: Optional DataFrame containing image information. Must contain at
            least the following columns : ``relative_path``, ``id``, ``width``,
            ``height``. Defaults to None

    Raises:
        ValueError: Errors when neither ``dataset_path`` nor ``data_file`` is specified

    Returns:
        Loaded dataset object
    """
    dataset_path, data_file = deduce_data_file_and_dataset_folder(
        dataset_path, data_file
    )

    if data_file.suffix in [".yml", ".yaml"]:
        return from_darknet_yolov5(
            dataset_path, data_file, ids_map=ids_map, image_info=image_info
        )

    if isinstance(ids_map, Path | str):
        with open(ids_map) as f:
            ids_map = json.load(f)
        assert isinstance(ids_map, dict)

    data_dict = open_data_file(data_file)
    train_file = data_dict["train"]
    valid_file = data_dict["valid"]
    names_file = data_dict["names"]
    assert isinstance(train_file, str)
    assert isinstance(valid_file, str)
    assert isinstance(names_file, str)
    # Filter is here to dismiss blank lines, especially at the end of the file
    with open(dataset_path / names_file) as f:
        name_list = filter(bool, f.read().split("\n"))
        dict(enumerate(name_list))

    if train_file:
        with open(dataset_path / train_file) as f:
            train_list = filter(bool, f.read().split("\n"))
    else:
        train_list = []
    with open(dataset_path / valid_file) as f:
        valid_list = filter(bool, f.read().split("\n"))

    dataset = from_darknet_generic(
        dataset_path,
        dataset_path,
        name_list,
        valid_list,
        "valid",
        ids_map=ids_map,
        image_info=image_info,
    )
    if train_list:
        dataset += from_darknet_generic(
            dataset_path,
            dataset_path,
            name_list,
            train_list,
            "train",
            ids_map=ids_map,
            image_info=image_info,
        )
    return dataset


def from_darknet_yolov5(
    dataset_path: Path | str | None = None,
    data_file: Path | str | None = None,
    splits: Iterable[str] | None = None,
    split_name_mapping: Mapping[str, str] | None = None,
    ids_map: dict[int, dict[str, Any]] | str | Path | None = None,
    image_info: pd.DataFrame | None = None,
) -> Dataset:
    """Creates dataset object from a darknet dataset. Note that category ids and
    image ids are not given in the dataset format and thus can only be sequential
    As such, if we want to convert the dataset back to another format that keeps track
    of image and category ids, we need to give image_info and class mapping from
    an external source. Here we expect it to be contained in a json label map for
    annotations and a DataFrame with similar columns as in the final dataset's images
    DataFrame for images

    Args:
        dataset_path: folder containing the dataset, from which the relative path
            are given. If not set, will use ``data_file``'s parent directory.
            Defaults to None
        data_file: data file containing info about names, lists of train and
            validation images. Must be a valid path to a Yaml file
            (either .yml or .yaml). If set to None, will use the file
            ``data.yaml`` at the root of ``dataset_path``. Defaults to None.
        splits: name of splits to load. if set to None, will consider every key in
            ``data_file`` that is neither ``names`` nor ``path`` to be a split.
            Every value in said keys must be either a valid folder or a valid text file,
            both relative to ``dataset_path``. Defaults to None.
        split_name_mapping: mapping dict to replace split names to other ones. split
            names not present in mapping will not be modified. If set to None,
            will apply yolov5 conventional mapping, i.e. 'val' -> 'valid' and
            'test' -> 'eval'. Defaults to None
        ids_map: Optional dictionary containing the id_remapping that was initially
            applied to create the darknet dataset. Will reverse it to get back to the
            original class mapping. The dictionary must have darknet dataset's category
            ids (in sequential order then) as keys and with corresponding values that
            are dictionaries containing ``name`` and ``id`` keys relative to this
            class. Note that this can also be a path to a json file containing the
            dictionary. Defaults to None.
        image_info: Optional DataFrame containing image information. Must contain at
            least the following columns : ``relative_path``, ``id``, ``width``,
            ``height``. Defaults to None

    Raises:
        ValueError: Errors when neither ``dataset_path`` nor ``data_file`` is specified

    Returns:
        Loaded dataset object
    """

    def get_folders_and_file(
        root_path: Path, data_path: str
    ) -> tuple[Path, list[str] | None]:
        full_path = root_path / data_path
        if full_path.is_dir():
            return full_path, None
        elif full_path.is_file():
            with open(full_path) as f:
                file_list = list(filter(bool, f.read().split("\n")))
            return root_path, file_list
        else:
            raise ValueError(f"Error, {full_path} is neither a folder nor a text file")

    if split_name_mapping is None:
        split_name_mapping = {"val": "valid", "test": "eval"}
    dataset_path, data_file = deduce_data_file_and_dataset_folder(
        dataset_path, data_file, default_data_file="data.yaml"
    )
    assert data_file.suffix in [".yml", ".yaml"]

    with open(data_file) as f:
        data_dict = yaml.safe_load(f)
    dataset_root = Path(data_dict.get("path", ""))
    if not dataset_root.is_absolute():
        dataset_root = dataset_path / dataset_root

    split_datasets = []
    if splits is None:
        splits = [
            name
            for name, value in data_dict.items()
            if name not in ["path", "names"] and isinstance(value, str)
        ]

    if not splits:
        raise ValueError("Not split was found or specified")

    for split_name in splits:
        split_path = data_dict[split_name]
        split_folder, split_files = get_folders_and_file(dataset_root, split_path)
        split_dataset = from_darknet_generic(
            split_folder,
            split_folder,
            data_dict["names"],
            split_files,
            split_name_mapping.get(split_name, split_name),
            ids_map,
            image_info,
        )
        split_datasets.append(split_dataset)
    return sum(split_datasets)  # pyright: ignore


def from_darknet_generic(
    images_root: Path | str,
    labels_root: Path | str,
    names: Iterable[str],
    image_files_list: Iterable[str | Path] | None = None,
    split: str | None = None,
    ids_map: dict[int, dict[str, Any]] | str | Path | None = None,
    image_info: pd.DataFrame | None = None,
) -> Dataset:
    """Generic function to load a darknet like dataset by only giving it folders,
    class names and optionally file list instead of a data file.

    Note:
        Unlike the darknet and yolov5 loaders, this function does only accept a single
        split value.

    Note:
        If no file list is given, the function will simply glob all image files in
        ``images_root`` folder.

    Args:
        images_root: Folder containing the image files to load
        labels_root: Folder containing the txt annotations files. Each annotation file
            is obtained by using the same relative path as the image equivalent and
            replacing the extension by txt. If no such file exist, it will not error
            but will assume there is no annotation
        names: list of classes used in the label map when saving this darknet dataset.
        image_files_list: list of images to read. Their path must be relative to
            ``images_root``. If set to None, will glob all files in ``images_root``
        split: Split name of the constructed dataset. Defaults to None.
        ids_map: Optional dictionary containing the id_remapping that was initially
            applied to create the darknet dataset. Will reverse it to get back to the
            original class mapping. The dictionary must have darknet dataset's category
            ids (in sequential order then) as keys and with corresponding values that
            are dictionaries containing ``name`` and ``id`` keys relative to this
            class. Note that this can also be a path to a json file containing the
            dictionary. Defaults to None.
        image_info: Optional DataFrame containing image information. Must contain at
            least the following columns : ``relative_path``, ``id``, ``width``,
            ``height``. Defaults to None

    Raises:
        FileNotFoundError: when the annotation file corresponding to a particular image
            file can not be found, be it using usual darknet convention or yolov5 one.

    Returns:
        Loaded dataset object
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    if image_files_list is None:
        image_files_list = get_images_from_folder(images_root)
    else:
        image_files_list = [Path(f) for f in image_files_list]
    images = []
    annotations = []

    if isinstance(ids_map, Path | str):
        with open(ids_map) as f:
            ids_map = json.load(f)
        assert isinstance(ids_map, dict)

    for image_number, image_path in enumerate(tqdm(image_files_list)):
        current_image = get_image_info(
            image_number=image_number,
            relative_path=image_path,
            absolute_path=images_root / image_path,
            image_info=image_info,
        )
        current_image["split"] = split
        images.append(current_image)

        annotation_path_simple = labels_root / image_path.with_suffix(".txt")
        annotation_path_yolov5 = labels_root / yolov5_img_path_to_label_path(image_path)

        if annotation_path_simple.is_file():
            annotation_path = annotation_path_simple
        elif annotation_path_yolov5.is_file():
            annotation_path = annotation_path_yolov5
        else:
            raise FileNotFoundError(
                f"Annotation could not be found. Neither {annotation_path_simple} nor"
                f" {annotation_path_yolov5} are files"
            )
        with open(annotation_path) as f:
            annotations_strings = filter(bool, f.read().split("\n"))
        annotations.extend(txt_to_bbox(annotations_strings, current_image["id"]))

    return to_dataset_object(
        images_root,
        dict(enumerate(names)),
        images,
        annotations,
        "cxwcyh",
        ids_map,
    )


def from_darknet_json(
    dataset_path: Path,
    json_path: Path,
    ids_map: dict[int, dict[str, Any]],
    image_info: pd.DataFrame | None,
    split_name: str = "eval",
) -> Dataset:
    """Same as from_darknet, expect the data file replaced with a json file containing
    directly annotations information. This is typically the format of predictions done
    by darknet's detector.

    Args:
        dataset_path: folder containing the dataset, from which
            the relative path are given
        json_path: json file containing a list of predictions as dictionaries. Each
            dictionary will have bbox info as well ad image path, which will be used
            to retrieve the original image id thanks to image_info DataFrame
        ids_map: dictionary containing the id_remapping that was initially applied to
            create the darknet dataset. Will reverse it to get back to the original
            class mapping. The dictionary must have darknet dataset's category ids
            (in sequential order then) as keys and with corresponding values that are
            dictionaries containing ``name`` and ``id`` keys relative to this class.
        image_info: DataFrame containing image information. Must contain at least the
            following columns : ``relative_path``, ``id``, ``width``, ``height``
        split_name: Name of the split that will be assigned to the ``split`` column of
            the resulting dataset's annotation dataframe.

    Returns:
        Loaded dataset object
    """
    images = []
    annotations = []
    with open(json_path) as f:
        json_list = json.load(f)
    for image_number, frame in enumerate(json_list):
        current_image = {}
        image_path = Path(frame["filename"])
        relative_path = get_relative_image_path(
            dataset_path=dataset_path, image_path=image_path
        )
        current_image["relative_path"] = relative_path
        current_image["type"] = image_path.suffix
        current_image["split"] = split_name
        absolute_path = dataset_path / relative_path
        current_image.update(
            **get_image_info(
                image_number=image_number,
                relative_path=relative_path,
                absolute_path=absolute_path,
                image_info=image_info,
            )
        )

        bboxes = []
        for current_object in frame["objects"]:
            coords = current_object["relative_coordinates"]
            bbox = {
                CX_COLUMN: coords["center_x"],
                CY_COLUMN: coords["center_y"],
                W_COLUMN: coords["width"],
                H_COLUMN: coords["height"],
                "category_id": current_object["class_id"],
                "category_str": current_object["name"],
                "confidence": current_object["confidence"],
                "image_id": current_image["id"],
            }
            bboxes.append(bbox)
        annotations.extend(bboxes)
        images.append(current_image)

    dataset = to_dataset_object(
        dataset_path,
        None,
        images,
        annotations,
        "cxwcyh",
        ids_map,
    )

    if image_info is not None:
        images_without_box = image_info[~image_info.index.isin(dataset.images.index)]
        dataset.images = pd.concat([image_info, images_without_box])
    dataset.images["split"] = split_name

    return dataset


def dataset_to_darknet(
    dataset: Dataset,
    output_path: Path | str,
    copy_images: bool = False,
    overwrite_images: bool = True,
    overwrite_labels: bool = True,
    yolo_version: int = 1,
    data_yaml_name: str = "data.yaml",
    split_name_mapping: dict[str, str] | None = None,
    create_split_folder: bool = False,
) -> None:
    """Save given dataset to darknet. Will recreate folder structure
    of dataset_path + image relative path. Will also create metadata files needed
    by darknet, i.e. names file, json label maps, list of train, validation and
    evaluation images (depending on available splits) and train and eval data files

    Args:
        dataset: dataset object to save as darknet
        output_path: root folder where images, corresponding text files
            describing bboxes will be saved along metadata files described above
        copy_images: If set to False, will create a symbolic link instead of copying.
            Much faster, but needs to keep original images in the same relative path.
            Defaults to False.
        overwrite_images: if set to False, will skip images that are already copied.
            Defaults to True.
        overwrite_labels: if set to False, will skip annotation that are already
            created. Defaults to True.
        yolo_version: version number for yolo. Officially supported versions are
            ``1`` to ``4`` (regular darknet), ``5`` (Ultralytics), ``7``
            (Chien-Yao Wang). Versions ``6`` and ``8`` are NOT officially supported, and
            will be assumed to all be version ``5``.
        data_yaml_name: if ``yolo_version`` is > 4, will be used as filename to save
            the yaml file at the root of ``output_path``
        split_name_mapping: mapping dict to replace split names to other ones. split
            names not present in mapping will not be modified. If set to None,
            will apply yolov5 conventional mapping,
            i.e. ``{'valid': 'val', 'validation': 'val', 'eval': 'test'}``.
            Defaults to None
        create_split_folder: if set to True, will create a dedicated folder for each
            split and will save images in it. Image paths in {split}.txt will be changed
            accordingly. Note that this changes the dataset structure. Defaults to False
    """
    try:
        assert_images_valid(dataset, load_images=False)
    except AssertionError as e:
        raise ValueError(
            "Dataset images are missing, check that the images root folder is the"
            " right one"
        ) from e
    if split_name_mapping is None:
        split_name_mapping = {"valid": "val", "validation": "val", "eval": "test"}

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    id_remapping = {
        i: {"id": cat_id, "name": cat_name}
        for i, (cat_id, cat_name) in enumerate(dataset.label_map.items())
    }

    # Remap classes so that category ids are now sequential
    sorted_dataset = dataset.remap_classes(
        {value["id"]: i for i, value in id_remapping.items()}
    )
    name_list = [*sorted_dataset.label_map.values()]
    name_list_path = output_path / "classes.names"
    with open(output_path / "classes.names", "w") as f:
        f.writelines(f"{name}\n" for name in name_list)
    with open(output_path / "ids_map.json", "w") as f:
        json.dump(id_remapping, f, indent=2)
    with open(output_path / "label_map.json", "w") as f:
        json.dump(dataset.label_map, f, indent=2)

    valid_splits = ["train", "valid", "eval"]

    if len(dataset.images["split"].isin(valid_splits)) < len(dataset.images):
        print(
            f"Darknet saver only takes these splits : {', '.join(valid_splits)}. "
            "Other values will be ignored"
        )

    remapped_splits = dataset.images["split"].replace(split_name_mapping)
    if create_split_folder:
        output_relative_path = pd.concat(
            [dataset.images["relative_path"], remapped_splits], axis=1
        ).apply(lambda x: x["split"] / x["relative_path"].with_suffix(".jpg"), axis=1)
    else:
        output_relative_path = dataset.images["relative_path"].apply(
            lambda x: x.with_suffix(".jpg")
        )
    split_path_lists = {
        split_name: path_list
        for split_name, path_list in output_relative_path.groupby(remapped_splits)
    }

    yaml_data = {
        "nc": len(name_list),
        "names": name_list,
    }
    for split_name, split_paths in split_path_lists.items():
        list_name = f"{split_name}.txt"
        list_path = output_path / list_name
        if yolo_version == 7:
            # For yolov7, paths need to be absolute, for both the txt file and the
            # image paths inside of it.
            # If possible, avoid saving to yolov7 format because it's hard to move
            # elsewhere afterwards

            # For the following pyright ignore, see the bug issue here
            # https://github.com/pandas-dev/pandas-stubs/issues/706
            # TODO : remove it when it is solved
            yaml_data[split_name] = str(list_path.absolute())  # pyright: ignore
            with open(list_path, "w") as f:
                f.writelines(
                    f"{output_path.absolute() / sample}\n" for sample in split_paths
                )
        else:
            yaml_data[split_name] = list_name  # pyright: ignore
            with open(list_path, "w") as f:
                f.writelines(f"{sample}\n" for sample in split_paths)

    if yolo_version <= 4:
        if len(split_path_lists.get("train", [])) > 0:  # pyright: ignore
            assert len(split_path_lists.get("val", [])) > 0  # pyright: ignore
        train_list_name = yaml_data["train"]
        val_list_name = yaml_data["val"]
        train_data_path = output_path / "train.data"
        write_data_file(
            {
                "classes": len(name_list),
                "train": train_list_name,
                "valid": val_list_name,
                "names": name_list_path.name,
            },
            train_data_path,
        )

        if len(split_path_lists.get("eval", [])) > 0:  # pyright: ignore
            eval_list_name = yaml_data["eval"]
            eval_data_path = output_path / "eval.data"
            write_data_file(
                {
                    "classes": len(name_list),
                    "valid": eval_list_name,
                    "names": name_list_path.name,
                },
                eval_data_path,
            )
    else:
        with open(output_path / data_yaml_name, "w") as f:
            yaml.safe_dump(yaml_data, f)

    bboxes_yolo = export_bbox(
        sorted_dataset.annotations, sorted_dataset.images, output_format="cxwcyh"
    )
    yolo_annotations = pd.concat([sorted_dataset.annotations, bboxes_yolo], axis=1)
    output_resolved = output_path.resolve()
    for image_id, image_data in tqdm(dataset.images.iterrows(), total=len(dataset)):
        instances = yolo_annotations[yolo_annotations["image_id"] == image_id]
        output_image_relative_path = output_relative_path.loc[
            image_id
        ]  # pyright: ignore
        input_image_path = (dataset.images_root / image_data["relative_path"]).resolve()
        output_image_path = output_resolved / output_image_relative_path
        output_image_path.parent.mkdir(exist_ok=True, parents=True)
        if yolo_version > 4:
            output_txt_path = yolov5_img_path_to_label_path(output_image_path)
        else:
            output_txt_path = output_image_path.with_suffix(".txt")
        if not output_image_path.is_file() or overwrite_images:
            if image_data["type"].lower() not in [".jpg", ".jpeg"]:
                image = imread(input_image_path)
                imwrite(output_image_path, image)
            elif copy_images:
                shutil.copy(input_image_path, output_image_path)
            else:
                output_image_path.unlink(missing_ok=True)
                output_image_path.symlink_to(
                    relpath(input_image_path, output_image_path.parent)
                )
        if not output_txt_path.is_file() or overwrite_labels:
            output_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_txt_path, "w") as f:
                f.writelines(bbox_to_txt(instances))
