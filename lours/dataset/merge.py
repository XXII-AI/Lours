from os.path import commonpath
from pathlib import Path

import pandas as pd

from lours.dataset import Dataset
from lours.utils.column_booleanizer import broadcast_booleanization
from lours.utils.label_map_merger import merge_label_maps
from lours.utils.testing import assert_frame_intersections_equal


def merge_datasets(
    dataset1: Dataset,
    dataset2: Dataset,
    allow_overlapping_image_ids: bool = True,
    realign_label_map: bool = False,
    ignore_index: bool = False,
    mark_origin: bool = False,
    overwrite_origin: bool = False,
) -> Dataset:
    """Merge two datasets and return a unique dataset object containing
    Samples from both. Result's images_root will be the common path of both
    datasets, and the image relative paths will be updated accordingly.
    Result's label map will be the superset of both label map,
    provided one is included in the other.

    Notes:
        - If possible, booleanized columns for images and annotations will be broadcast
          together. See :func:`lours.utils.column_booleanizer.broadcast_booleanization`
        - If one of the dataset has an absolute path as ``images_root``, the other
          dataset images root path will also be converted to absolute.
        - If both datasets have the same name, the output will have the same name
          as well.
        - If datasets have a different name, the output will have the concatenation
          of both names separate by a "+" sign. The merge output of "A" and "B" will
          be thus names "A+B".
        - If one dataset has no name (``dataset.name`` is ``None``), the output will
          take the name of the other.
        - If ``mark_origin`` is selected, it will be effective only if datasets have
          different actual names (not ``None``)

    Args:
        dataset1: First dataset to merge.
        dataset2: Second dataset to merge with dataset1. This dataset must be
            compatible with the first one, i.e. one label map is included with the
            other,  image and annotation ids are mutually exclusives between
            datasets (unless `ignore_index` is False), and booleanized columns are
            compatible with each other.
        allow_overlapping_image_ids: if set to True, will try to join images
            dataframes with overlapping ids. The whole rows (i.e. with values from
            columns present in both dataframes) must match, as well as
            the images_root. In that case, annotations with this image_id
            (from self or other) will be assumed to come from the same image.
            Defaults to True
        realign_label_map: If set to True, will try to remap classes of dataset2 to
            match the label map fo dataset1, to avoid a potential error due to
            incompatible label maps.
        ignore_index: if set to True, will ignore overlapping ids
            for images and annotations and reset them. Will update the ``image_id``
            column in the annotations accordingly. Note that this option makes the
            former option useless. Defaults to False.
        mark_origin: If set to True, and if both datasets have a different name, will
            add two columns "origina_dataset_name" and "origin" for images and
            annotations dataframes, indicating respectively the name
            of the origin dataset, and its id in the original dataset. Defaults to True.
        overwrite_origin: If set to True, will overwrite already existing columns in
            input datasets dataframes. Otherwise, will only mark origin if it's not
            present. Defaults to False.

    Raises:
        ValueError: Error if the two datasets are incompatible (see above)

    Returns:
        Merged dataset.
    """
    needs_new_name = (
        dataset1.dataset_name is not None
        and dataset2.dataset_name is not None
        and dataset2.dataset_name != dataset1.dataset_name
    )

    def mark_origin_to_dataset(dataset: Dataset) -> Dataset:
        dataset_columns = set(dataset.images.columns) & set(dataset.annotations.columns)
        if (
            "origin" in dataset_columns
            and "origin_id" in dataset_columns
            and not overwrite_origin
        ):
            return dataset
        return dataset.from_template(
            images=dataset.images.assign(
                origin=dataset.dataset_name,
                origin_id=dataset.images.index,
            ),
            annotations=dataset.annotations.assign(
                origin=dataset.dataset_name,
                origin_id=dataset.annotations.index,
            ),
        )

    if mark_origin and needs_new_name:
        dataset1 = mark_origin_to_dataset(dataset1)
        dataset2 = mark_origin_to_dataset(dataset2)

    if ignore_index:
        return merge_datasets(
            dataset1.reset_index(),
            dataset2.reset_index(len(dataset1)),
            allow_overlapping_image_ids=True,
            ignore_index=False,
        )

    # images_root to grab images might not be the same
    # Get the common path and update images relative paths to
    # be relative to that new path
    if dataset1.images_root != dataset2.images_root:
        if dataset1.images_root.is_absolute() or dataset2.images_root.is_absolute():
            images_root = commonpath(
                [
                    dataset1.images_root.absolute(),
                    dataset2.images_root.absolute(),
                ]
            )
        else:
            images_root = commonpath([dataset1.images_root, dataset2.images_root])
        images_root = Path(images_root)
        dataset1 = dataset1.reset_images_root(images_root)
        dataset2 = dataset2.reset_images_root(images_root)
    else:
        images_root = dataset1.images_root

    if realign_label_map:
        dataset2 = dataset2.remap_from_other(dataset1)
        label_map = merge_label_maps(
            dataset1.label_map, dataset2.label_map, method="outer"
        )
    else:
        label_map = merge_label_maps(
            dataset1.label_map, dataset2.label_map, method="outer"
        )

    dataset1_images, dataset2_images, booleanized_image_columns = (
        broadcast_booleanization(
            dataset1.images,
            dataset2.images,
            booleanized_columns1=dataset1.booleanized_columns["images"],
            booleanized_columns2=dataset2.booleanized_columns["images"],
        )
    )
    dataset1_annotations, dataset2_annotations, booleanized_annotations_columns = (
        broadcast_booleanization(
            dataset1.annotations,
            dataset2.annotations,
            booleanized_columns1=dataset1.booleanized_columns["annotations"],
            booleanized_columns2=dataset2.booleanized_columns["annotations"],
        )
    )

    dataset1_images_ids = set(dataset1.images.index)
    dataset2_images_ids = set(dataset2.images.index)
    mutual_images_ids = dataset1_images_ids & dataset2_images_ids
    dataset1_images_columns = set(dataset1.images.columns)
    dataset2_images_columns = set(dataset2.images.columns)
    mutual_images_columns = dataset1_images_columns & dataset2_images_columns

    if mutual_images_ids and not allow_overlapping_image_ids:
        raise ValueError(
            "Overlapping image ids not permitted. Consider using the"
            " allow_overlapping_image_ids or ignore_index options"
        )

    assert_frame_intersections_equal(
        dataset1_images.drop(["origin", "origin_id"], axis=1, errors="ignore"),
        dataset2_images.drop(["origin", "origin_id"], axis=1, errors="ignore"),
    )

    # Concat horizontally by extending images from dataset1 with columns from dataset2
    # and then vertically by extending images with dataset2 images which id is not
    # in dataset1 images index.
    dataset1_images = dataset1_images.join(
        dataset2_images.loc[
            list(mutual_images_ids),
            list(dataset2_images_columns - mutual_images_columns),
        ]
    )
    dataset2_images = dataset2_images.loc[
        list(dataset2_images_ids - mutual_images_ids), :
    ]

    images = pd.concat([dataset1_images, dataset2_images])

    # Merge annotations.
    mutual_instance_ids = set(dataset1_annotations.index).intersection(
        set(dataset2_annotations.index)
    )
    # Only reset index of dataset2's annotations if there is overlap.
    # However, keep the index of first dataset as is
    if mutual_instance_ids:
        dataset2_annotations.index += (
            dataset1_annotations.index.max() - dataset2_annotations.index.min() + 1
        )
    annotations = pd.concat([dataset1_annotations, dataset2_annotations])
    annotations.index.name = "id"

    if needs_new_name:
        output_dataset_name = f"{dataset1.dataset_name}+{dataset2.dataset_name}"
    elif dataset2.dataset_name is None:
        output_dataset_name = dataset1.dataset_name
    else:
        output_dataset_name = dataset2.dataset_name
    output = Dataset(
        images_root=images_root,
        images=images,
        annotations=annotations,
        label_map=label_map,
        dataset_name=output_dataset_name,
    )
    output.booleanized_columns["images"] = booleanized_image_columns
    output.booleanized_columns["annotations"] = booleanized_annotations_columns

    return output
