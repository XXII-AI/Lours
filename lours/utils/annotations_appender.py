from collections import defaultdict
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy import ndarray

from ..dataset import Dataset
from . import BBOX_COLUMN_NAMES
from .bbox_converter import import_bbox
from .label_map_merger import merge_label_maps

*_, WIDTH_NAME, HEIGHT_NAME = BBOX_COLUMN_NAMES

D = TypeVar("D", bound=Dataset)
Scalar = float | str | bool
ScalarSequence = Sequence[float] | Sequence[str] | Sequence[bool]


def broadcast_annotations(
    image_id: int | Sequence[int] | ndarray,
    bbox_coordinates: ndarray | Sequence[float] | Sequence[Sequence[float]],
    category_id: int | ndarray | Sequence[int],
    confidence: float | Sequence[float] | ndarray | None = None,
    **other: ndarray | Scalar | ScalarSequence,
) -> dict[str, np.ndarray]:
    """Broadcast together detection annotation attributes.

    Every attribute except ``bbox_coordinates`` can be only 1 element, and will be
    duplicated to match the length of ``bbox_coordinates``.

    If its more than 1 element, the length must match the length of ``bbox_coordinates``

    Args:
        image_id: id or 1D array of shapes [N] of ids of image corresponding to each
            annotation to be appended
        bbox_coordinates: 2D numpy array of shape [N, 4] or [N, 2] (for a keypoint)
            corresponding to the bounding box coordinates.
            Can also be of the shape [4] or [2] if there is only one
            bounding box/keypoint
        category_id: int value or 1D numpy array of shape [N] corresponding to
            the category of each bounding box
        confidence: when dealing with the special case of predictions, 1D numpy
            array of shape [N] corresponding to the confidence of the prediction.
            None otherwise. Defaults to None.
        **other: Other possible fields to the annotation

    Raises:
        ValueError: Will be raised if the bbox_coordinates shape is not [N, 4] [N, 2] or
            [2] or [4]
        ValueError: Will be raised if the values to broadcast are not scalar or array of
            size [1] or size [N]

    Returns:
        dictionary with the annotation attribute name as keys
        ("image_id", "category_id", etc) and the broadcast of values as numpy arrays
        as values.
    """
    if not isinstance(bbox_coordinates, ndarray):
        bbox_coordinates = np.array(bbox_coordinates)
    bbox_shape = bbox_coordinates.shape
    if len(bbox_shape) == 1:
        bbox_coordinates = bbox_coordinates[None]
    if (len(bbox_coordinates.shape) != 2) or (bbox_coordinates.shape[1] not in [2, 4]):
        raise ValueError(
            "Error for bbox_coordinates, expected shape of form [4], [2], [N, 4]"
            f" or [N, 2] got {bbox_shape}"
        )
    n_annot = bbox_coordinates.shape[0]

    def broadcast_scalar(
        value_to_broadcast: Scalar | ScalarSequence | np.ndarray,
        key_name: str,
    ) -> np.ndarray:
        if np.ndim(value_to_broadcast) == 0:
            broadcasted = np.tile(value_to_broadcast, n_annot)
        elif np.ndim(value_to_broadcast) == 1:
            value_to_broadcast_np = np.array(value_to_broadcast)
            if value_to_broadcast_np.shape[0] == 1:
                broadcasted = np.tile(value_to_broadcast_np, n_annot)
            elif value_to_broadcast_np.shape[0] == n_annot:
                broadcasted = value_to_broadcast_np
            else:
                broadcasted = None
        else:
            broadcasted = None
        if broadcasted is None:
            raise ValueError(
                f"Value to broadcast {key_name} can only be either a scalr of a 1D"
                f" dimensional array of length 1 or {n_annot}"
            )
        return broadcasted

    result = {}
    result["image_id"] = broadcast_scalar(image_id, "image_id")
    result["bbox_coordinates"] = bbox_coordinates
    result["category_id"] = broadcast_scalar(category_id, "category_id")
    if confidence is not None:
        if n_annot == 1:
            result["confidence"] = broadcast_scalar(confidence, "confidence")
        else:
            confidence = np.array(confidence)
            if confidence.shape != (n_annot,):
                raise ValueError(
                    "Confidence field can only be the same length as bounding boxes."
                    f" Expected length of {n_annot}, got shape of {confidence.shape}"
                )
            result["confidence"] = confidence

    for key, value in other.items():
        result[key] = broadcast_scalar(value, key)

    return result


def add_detection_annotation(
    input_dataset: D,
    image_id: int | Sequence[int] | ndarray,
    bbox_coordinates: Sequence[float] | Sequence[Sequence[float]] | ndarray,
    format_string: str,
    category_id: ndarray | int | Sequence[int],
    inplace: bool = False,
    label_map: dict[int, str] | None = None,
    category_ids_mapping: dict[int, int] | None = None,
    confidence: float | ndarray | Sequence[float] | None = None,
    **other_columns: Scalar | ScalarSequence | ndarray,
) -> D:
    """Add one or multiple detection annotations to the current dataset.
    In the case of a single annotation, every option can be a single value, but in the
    case of multiple annotations, every option needs to be an array of such values, and
    every array needs to be the same length.

    Note:
        In additions to the following options, you can add other fields as well, with
        keyword arguments.


    Args:
        input_dataset: Dataset to which we want to add new annotations
        image_id: image identifier to link each detection to the corresponding image
        bbox_coordinates: list of coordinates for the bounding box. Can follow any
            compatible format, as long as it is given in the next format
        format_string: format of coordinates, whether coordinates are relatives, using
            corner points of the box, box dimensions, etc. See :func:`.import_bbox` for
            more info
        category_id: category of each detection. Label will be deduced from dataset's
            label map
        inplace: if set to True, will modify the dataset in place and return itself.
            Else, will return a modified Dataset. Defaults to False.
        label_map: In the case the current dataset's label map is incomplete, merge it
            with this new label map. current label map and new label map must be
            compatible, see :func:`.merge_label_maps`. Defaults to None.
        category_ids_mapping: Optional dictionary to map annotated category ids into the
            right ids. This is useful for example when a neural network can only use a
            contiguous label map.
        confidence: Optional field for confidence, in the case annotations are actually
            predictions. Must the same length as bbox_coordinates. In the case of a
            single prediction, can also be a float. Defaults to None.
        **other_columns: Other column representing custom fields.

    Raises:
        ValueError: raised when giving numpy arrays are not the same number of elements,
            or if the bounding box coordinates is not of the shape either [4], or [N, 4]

    Returns:
        If ``inplace`` is False, new dataset object with appended annotations.
        Otherwise, the updated ``input_dataset``.
    """
    if label_map is not None:
        label_map = merge_label_maps(input_dataset.label_map, label_map, method="outer")
    else:
        label_map = input_dataset.label_map

    columns = broadcast_annotations(
        image_id=image_id,
        bbox_coordinates=bbox_coordinates,
        category_id=category_id,
        confidence=confidence,
        **other_columns,
    )
    bbox_coordinates = columns.pop("bbox_coordinates")
    for name, column in columns.items():
        if column.ndim != 1:
            raise ValueError(
                "Field as arrays can only be 1-dimensional, but got a shape of"
                f" {column.shape} for column {name}"
            )

    if bbox_coordinates.shape[1] not in [2, 4]:
        raise ValueError(
            "Error for bbox_coordinates, expected an array of 4 or 2 columns, got"
            f" shape of {bbox_coordinates.shape}"
        )
    annotations_df = pd.DataFrame(columns)
    if input_dataset.len_annot() > 0:
        first_index = input_dataset.annotations.index.max() + 1
        annotations_df.index += first_index
    if category_ids_mapping is not None:
        annotations_df["category_id"] = annotations_df["category_id"].replace(
            category_ids_mapping
        )

    bbox_df = import_bbox(
        bounding_boxes=bbox_coordinates,
        images_df=input_dataset.images,
        image_ids=annotations_df["image_id"],
        input_format=format_string,
    )

    bbox_df.index = annotations_df.index

    annotations_df = pd.concat([annotations_df, bbox_df], axis=1)

    if ("area" in input_dataset.annotations.columns) and (
        "area" not in annotations_df.columns
    ):
        annotations_df["area"] = bbox_df[WIDTH_NAME] * bbox_df[HEIGHT_NAME]

    annotations_df = pd.concat([input_dataset.annotations, annotations_df])

    if inplace:
        input_dataset.label_map = label_map
        input_dataset.annotations = annotations_df
        input_dataset.init_annotations()
        return input_dataset
    else:
        return input_dataset.from_template(
            annotations=annotations_df, label_map=label_map
        )


class AnnotationAppender:
    """Context manager to easily add detection tensors, as if the Dataset was a list
    after the appending is finished, the appender construct big numpy arrays to
    concatenate to the dataset's annotations dataframe
    """

    def __init__(
        self,
        dataset: Dataset,
        format_string: str = "XYWH",
        category_ids_mapping: dict[int, int] | None = None,
        label_map: dict[int, str] | None = None,
    ):
        """Main constructor.

        Args:
            dataset: dataset the annotations will be appended to.
            format_string: String describing bounding box format. Defaults to "XYWH".
            category_ids_mapping: Optional dictionary to map annotated category ids into
                the right ids. This is useful for example when a neural network can only
                use a contiguous label map.
            label_map: Optional dictionary to provide ``category_id -> ``category_str``
                mapping for additional categories in appended annotations.
                Defaults to None.
        """
        self.dataset = dataset
        self.image_ids = set(dataset.images.index)
        self.format_string = format_string
        self.label_map = label_map
        self.categroy_ids_mapping = category_ids_mapping
        self.reset()

    def reset(self):
        """Creates an empty dictionary that will be fed new annotations."""
        self.annotations_to_append = defaultdict(dict)
        self.i = 0
        self.index = {}

    def __enter__(self):
        """Function called at the beginning of ``with`` context.

        Returns:
            self class, with the ``append``method.
        """
        self.reset()
        return self

    def append(
        self,
        image_id: int | Sequence[int] | ndarray,
        bbox_coordinates: ndarray | Sequence[float],
        category_id: int | Sequence[int] | ndarray,
        confidence: ndarray | float | Sequence[float] | None = None,
        **other: Scalar | ScalarSequence | ndarray,
    ) -> None:
        """Append annotations for a particular image id. Everything except image id
        is expected to be a numpy array. Note that in addition to the regular bounding
        boxes coordinates, category and confidence, you can add other fields as long as
        they are numpy array of the same length. If no column exist in the dataset's
        annotations dataframe, it will be created (setting the already existing
        annotations to NaN for this column)

        Args:
            image_id: id or 1D array of shapes [N] of ids of image corresponding to each
                annotation to be appended
            bbox_coordinates: 2D numpy array of shape [N, 4] corresponding to the
                bounding box coordinates
            category_id: hashable value or 1D numpy array of shape [N] corresponding to
                the category of each bounding box
            confidence: when dealing with the special case of predictions, 1D numpy
                array of shape [N] corresponding to the confidence of the prediction,
                which can also be a float if N == 1. None otherwise. Defaults to None.
            **other: Other possible fields to the annotation

        Raises:
            ValueError: raised when the given numpy arrays are not of the same length,
                or if bounding box coordinates are not a 2D array with 4 columns.
            ValueError: raised when the given image id is not present in the dataset's
                image dataframe
        """
        ids_to_check = [image_id] if isinstance(image_id, int) else image_id
        for i in ids_to_check:
            if i not in self.image_ids:
                raise ValueError(f"Image id {i} is not in dataset's images dataframe")
        broadcasted_annotations = broadcast_annotations(
            image_id=image_id,
            bbox_coordinates=bbox_coordinates,
            category_id=category_id,
            confidence=confidence,
            **other,
        )
        n_annot = len(broadcasted_annotations["image_id"])
        # Note that we need index because not all append calls have the same keywords
        # As such, the index can help up know which rows should have actual value,
        # and which rows should have pd.NA
        self.index[self.i] = np.arange(self.i, self.i + n_annot)

        for key, value in broadcasted_annotations.items():
            self.annotations_to_append[key][self.i] = value
        self.i += n_annot

    def finish(self) -> None:
        """Concatenate all annotations given by ``append`` method into one dataframe,
        with the right bounding box format and concatenate it to the Dataset's
        annotation original DataFrame.
        """
        concatenated_annotations = []
        try:
            bbox = np.concatenate(
                list(self.annotations_to_append.pop("bbox_coordinates").values())
            )
        except KeyError:
            # No bbox coordinates were added
            return
        for name, annot in self.annotations_to_append.items():
            index = np.concatenate([self.index[k] for k in annot.keys()])
            array = np.concatenate(list(annot.values()))
            concatenated_annotations.append(pd.Series(array, index=index, name=name))
        annotations_to_append = pd.concat(concatenated_annotations, axis=1)

        annotations_to_append_dict = {
            str(k): annotations_to_append[k].to_numpy()
            for k in annotations_to_append.columns
        }
        add_detection_annotation(
            self.dataset,
            format_string=self.format_string,
            inplace=True,
            label_map=self.label_map,
            category_ids_mapping=self.categroy_ids_mapping,
            bbox_coordinates=bbox,
            **annotations_to_append_dict,
        )

    def __exit__(self, exit_type: Any, exit_value: Any, traceback: Any) -> None:
        """Function called at the en of context, when annotations have been appended.

        Args:
            exit_type: For context manager compatibility. Not used here.
            exit_value: For context manager compatibility. Not used here.
            traceback: For context manager compatibility. Not used here.
        """
        self.finish()
