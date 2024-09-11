from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd
from tqdm.auto import tqdm

from . import bbox_converter, try_import_fiftyone

if TYPE_CHECKING:
    import fiftyone as fo
else:
    fo = try_import_fiftyone()


def make_fiftyone_compatible(
    input_df: pd.DataFrame,
    column_names: Sequence[str] = (),
    replacement_string: str = "->",
) -> tuple[pd.DataFrame, list[str]]:
    """Make column names compatible with fiftyone.

    Fiftyone is incompatible with names with a ".", so replace them with a proper
    character.

    Fiftyone is also incompatible with names starting with 'attributes', which is the
    case for libia.Dataset.annotations attributes columns, so we replace the string
    'attributes' with 'attr' in each column name.

    Note:
        If no name in ``column_names`` has a forbidden character, this function simply
        return its inputs.

    Args:
        input_df: DataFrame for which column names will be replaced
        column_names: Column names to rename. If the names are not present in the
            ``input``, no error will be raised. Defaults to ().
        replacement_string: string used to replace forbidden characters.
            Defaults to ``->``.

    Returns:
        tuple with 2 elements
         - DataFrame with modified column names from ``input``
         - List of modified names.

    """
    if not column_names:
        return input_df, []
    new_column_names = [name.replace(".", replacement_string) for name in column_names]
    # replace column names containing 'attributes' with 'attr' otherwise the fiftyone
    # web app will crash if we try to filter on these columns
    # e.g. "attributes->out_of_frame" will become "attr->out_of_frame" in fiftyone
    new_column_names = [name.replace("attributes", "attr") for name in new_column_names]
    output = input_df.rename(dict(zip(column_names, new_column_names)), axis=1)
    return output, new_column_names


def annotations_to_fiftyone(
    annotations_frame: pd.DataFrame,
    attribute_columns: Sequence[str] = (),
    bbox_column: str = "bbox",
    allow_keypoints: bool = True,
) -> pd.DataFrame:
    """Convert annotations frame into a DataFrame using the same index with
    :class:`~fiftyone.core.labels.Detection` or
    :class:`~fiftyone.core.labels.Keypoint` object in the "fo_detection" column and
    other additional fiftyone columns.

    Args:
        annotations_frame: DataFrame containing information about annotation. Most
            likely coming out of a :class:`.Dataset` object.
        attribute_columns: Columns describing real attributes. these attributes are
            differentiated from functional metadata such as ``image_id``.
            Defaults to ().
        bbox_column: column containing bounding coordinates lists. Must be already
            converted to fiftyone's xywh format. Defaults to "bbox".
        allow_keypoints: If set to True, will deduce keypoints from bounding box of size
            0. If not, every bounding box will be a detection. Defaults to True.

    Returns:
        DataFrame with the same index as ``annotations_frame``, containing
        ``fo_detection``, ``fo_id`` and ``is_keypoint`` columns.

        - ``fo_detection`` is the fiftyone object describing the detection, to be added
          to the related image sample. It can be either a
          :class:`~fiftyone.core.labels.Detection`
          or :class:`~fiftyone.core.labels.Keypoint` object
        - ``fo_id`` if the UUID given by fiftyone to identify the detection in the
          database.
        - ``is_keypoint`` is the boolean value indicating if the fiftyone object in the
          former column is a :class:`~fiftyone.core.labels.Detection` or
          :class:`~fiftyone.core.labels.Keypoint` object.
          This will make filtering much faster than looking up the type for each row.
    """

    def to_detection(row: pd.Series) -> fo.Detection:
        return fo.Detection(
            label=row["category_str"],
            label_id=row["category_id"],
            lours_id=row.name,
            bounding_box=row[bbox_column],
            **row[attribute_columns].dropna(),  # pyright: ignore
        )

    def to_keypoint(row: pd.Series) -> fo.Keypoint:
        return fo.Keypoint(
            label=row["category_str"],
            label_id=row["category_id"],
            lours_id=row.name,
            points=[row[bbox_column][:2]],
            **row[attribute_columns].dropna(),  # pyright: ignore
        )

    if allow_keypoints:
        is_keypoint = (annotations_frame["box_width"] == 0) & (
            annotations_frame["box_height"] == 0
        )
        if (~is_keypoint).sum() > 0:
            detections = annotations_frame[~is_keypoint].apply(
                to_detection,  # pyright: ignore
                axis=1,
            )
        else:
            detections = pd.Series([])
        keypoints = annotations_frame[is_keypoint].apply(
            to_keypoint,  # pyright: ignore
            axis=1,
        )
        detections = pd.DataFrame(
            {
                "is_keypoint": is_keypoint,
                "fo_detection": pd.concat([detections, keypoints]),
            }
        )
    else:
        detections = (
            annotations_frame.apply(to_detection, axis=1)  # pyright: ignore
            .rename("fo_detection")
            .to_frame()
        )
        detections["is_keypoint"] = False
    detections["fo_id"] = detections["fo_detection"].apply(lambda x: x.id)
    return detections


def create_fo_dataset(
    name: str,
    images_root: Path,
    images: pd.DataFrame,
    annotations: dict[str, pd.DataFrame],
    bounding_box_formats: dict[str, str] | None = None,
    label_map: dict[int, str] | None = None,
    image_tag_columns: Sequence[str] = (),
    annotations_attributes_columns: Sequence[str] | dict[str, Sequence[str]] = (),
    allow_keypoints: bool = False,
    existing: Literal["error", "update", "erase"] = "error",
) -> tuple[fo.Dataset, pd.Series, dict[str, pd.DataFrame]]:
    """Generic function to create a fiftyone dataset from images and annotations
    DataFrames. See :func:`.dataset_to_fiftyone` and :func:`.evaluator_to_fityone`
    for more specific functions

    Args:
        name: Name of the fiftyone dataset to add the samples to. If the dataset
            does not exist, it will be created.
        images_root: root folder for images. Fiftyone will try to load the images by
            concatenating this path and the value in images' ``relative_path`` column.
        images: DataFrame comprising image information.
            Must have at least ``relative_path`` column, and the column specified in
            ``image_tag_columns``
        annotations: dictionary of DataFrames comprising detections annotations
            information. Each entry must have at least ``image_id``, `category_id``,
            ``category_str`` (if ``label_map`` argument is None) columns,
            and the compatible columns for bbox conversion given the corresponding
            input format (see :func:`.convert_bbox`)
        bounding_box_formats: dictionary of format strings to convert bounding boxes
            of given annotations.
            For each dictionary entry in ``annotations`` dictionary, if its key is
            not included in this dictionary, the format "XYWH" (COCO cAIpy) will be
            assumed. If set to None, the format "XYWH" will always be assumed.
            Defaults to None
        label_map: dictionary comprising the category id -> category string
            correspondence, similar to :class:`.Dataset` and :class:`.Evaluator`
            label maps. If given, will populate the ``category_str`` of each annotation
            DataFrame. If set to None, will assume the column is already present.
            Defaults to None.
        image_tag_columns: List of column names to use for sample attributes in given
            ``images`` DataFrame when creating fiftyone samples. Defaults to ().
        annotations_attributes_columns: Either List of column names or dictionary of
            lists of column names. Is used to give attributes to detection which are
            then added to the created. If it's a dictionary, each annotation set in the
            ``annotations`` dictionary gets its own list of columns to use as
            attributes. If not, the ame list of column will be used for all annotations.
            Defaults to ().
        allow_keypoints: if set to True, will convert bounding boxes of size 0, 0 to
            keypoints
        existing: What to do in case there is already a fiftyone dataset with the
            same name.

            - "error": will raise an error.
            - "erase": will erase the existing dataset before uploading
                this one
            - "update": will try to update the dataset by fusing together samples
                with the same "relative_path"

            Defaults to "error".

    Returns:
        tuple with three elements
         - Fiftyone dataset that can then be used to launch the webapp with
           :func:`fiftyone.launch_app`
         - Series with the same index as ``images`` input dataframe, containing the
           fiftyone index of each image's corresponding sample. Useful when the image
           needs to be modified.
         - Dictionary of Dataframes, with the same keys as ``annotations`` input
           dictionary. Each value is a DataFrame with the same index as its
           corresponding value in ``annotations``. Its columns are ``fo_id`` and
           ``is_keypoint``

           - ``fo_id`` is the fiftyone index of each annotation (whether it is a
             :class:`~fiftyone.core.labels.Keypoint` or a
             :class:`~fiftyone.core.labels.Detection`)
           - ``is_keypoint`` is a bool column indicating if the annotation is a
             :class:`~fiftyone.core.labels.Keypoint` object or simply a
             :class:`~fiftyone.core.labels.Detection` object.
    """
    do_update = False
    if fo.dataset_exists(name):
        if existing == "update":
            result = fo.load_dataset(name)
            do_update = True
        elif existing == "erase":
            fo.delete_dataset(name)
            result = fo.Dataset(name)
        else:
            raise FileExistsError("Dataset already exists")
    else:
        result = fo.Dataset(name)
    # Manual casting to a dataset because the fiftyone Dataset constructor is
    # not compliant with standard metaclasses
    # see https://github.com/microsoft/pyright/discussions/5583
    result = cast(fo.Dataset, result)
    if bounding_box_formats is None:
        bounding_box_formats = {}
    if not isinstance(annotations_attributes_columns, dict):
        annotations_attributes_columns = {
            name: annotations_attributes_columns for name in annotations
        }

    images, image_tag_columns = make_fiftyone_compatible(images, image_tag_columns)
    for name in list(annotations.keys()):
        annotations[name], annotations_attributes_columns[name] = (
            make_fiftyone_compatible(
                annotations[name], annotations_attributes_columns[name]
            )
        )

    bbox_column_names = bbox_converter.column_names_from_format_string("xywh")
    fo_detections = {}
    for annotations_name, annotations_frame in annotations.items():
        if len(annotations_frame) == 0:
            continue
        input_format = bounding_box_formats.get(annotations_name, "XYWH")
        bbox = bbox_converter.convert_bbox(
            annotations_frame, images, input_format=input_format, output_format="xywh"
        )

        bbox = pd.Series(
            bbox[bbox_column_names].to_numpy().tolist(),
            index=annotations_frame.index,
            name="bbox",
        ).to_frame()
        to_concat = [annotations_frame, bbox]
        if label_map is not None and "category_str" not in annotations_frame.columns:
            to_concat.append(
                annotations_frame["category_id"]
                .replace(label_map)
                .rename("category_str")
                .to_frame()
            )

        current_converted_annotations = pd.concat(to_concat, axis=1)
        current_fo_detections = annotations_to_fiftyone(
            current_converted_annotations,
            annotations_attributes_columns.get(annotations_name, []),
            bbox_column="bbox",
            allow_keypoints=allow_keypoints,
        )
        current_fo_detections["image_id"] = annotations_frame["image_id"]
        fo_detections[annotations_name] = current_fo_detections

    samples = {}
    sample_ids = pd.Series(
        pd.NA,  # pyright: ignore
        index=images.index,
        name="sample_id",
    )
    detections_fo_metadata = {
        annotation_name: fo_detections_frame[["fo_id", "is_keypoint"]]
        for annotation_name, fo_detections_frame in fo_detections.items()
    }

    for image_id, image_data in tqdm(images.iterrows(), total=len(images)):
        image_path = images_root / image_data["relative_path"]
        metadata = fo.ImageMetadata(
            width=image_data["width"], height=image_data["height"]
        )
        sample = fo.Sample(
            filepath=image_path,
            lours_id=image_id,
            relative_path=str(image_data["relative_path"]),
            metadata=metadata,
            split=image_data.get("split", None),
            **image_data.loc[image_tag_columns].dropna(),  # pyright: ignore
        )
        samples[image_id] = sample
        sample_ids[image_id] = sample.id

    for annotation_name, fo_detections_frame in fo_detections.items():
        for (image_id, is_keypoint), detections_subset in fo_detections_frame.groupby(
            [
                "image_id",
                "is_keypoint",
            ]
        )["fo_detection"]:
            if is_keypoint:
                image_detections = fo.Keypoints(keypoints=detections_subset.to_list())
            else:
                image_detections = fo.Detections(detections=detections_subset.to_list())
            attribute_name = (
                f"{annotation_name}_{'keypoint' if is_keypoint else 'detection'}"
            )
            samples[image_id][attribute_name] = image_detections

    if do_update:
        result.merge_samples(samples.values(), key_field="relative_path", dynamic=True)
    else:
        result.add_samples(samples.values(), dynamic=True)
    result.add_dynamic_sample_fields()
    result.save()

    return (result, sample_ids, detections_fo_metadata)
