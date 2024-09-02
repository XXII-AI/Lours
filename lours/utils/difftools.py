"""Set of tools to differentiate datasets or evaluators"""

from collections.abc import Iterable
from warnings import warn

from lours.dataset import Dataset
from lours.utils.label_map_merger import IncompatibleLabelMapsError, merge_label_maps


def dataset_diff(
    left_dataset: Dataset,
    right_dataset: Dataset,
    exclude_image_columns: Iterable[str] = (),
    exclude_annotations_columns: Iterable[str] = (),
) -> tuple[Dataset, Dataset, Dataset]:
    """Differentiate two datasets and construct the difference datasets, only containing
    elements that are in one of the two datasets but not the other

    this function outputs the differences with 2 datasets that are constructed with
    images and annotations specific to each dataset and a third dataset with common
    images and annotations.

    As such, you should theoretically be able to reconstruct the left dataset with
    the first difference dataset and the common dataset, and reconstruct the right
    dataset with the second difference dataset and the common dataset.

    Note:
        if one dataset has a column in its dataframes the other dataset doesn't have,
        and that column is not included in ``exclude_image_columns`` or
        ``exclude_annotations_column``, the dataframes and thus the datasets will be
        considered entirely different, and the common dataset will be empty

    Note:
        if ``exclude_image_columns`` or ``exclude_annotations_columns`` is not empty,
        it is not guaranteed to be able to reconstruct left or right dataset with common
        datasets and difference datasets, only the datasets minus the excluded columns.


    Args:
        left_dataset: left dataset to compare
        right_dataset: right dataset to compare
        exclude_image_columns: list of names of columns to ignore in image dataframes
            for the comparison.
        exclude_annotations_columns: list of names of columns to ignore in annotations
            dataframes for the comparison.

    Returns:
        tuple with 3 datasets
         - dataset with images and annotations that are specific to ``left_dataset``
         - dataset with images and annotations that are specific to ``right_dataset``
         - dataset with images and annotations that are common to both input datasets.
    """
    try:
        merge_label_maps(left_dataset.label_map, right_dataset.label_map)
    except IncompatibleLabelMapsError:
        warn("Incompatible label maps, dataset cannot have mutual info", RuntimeWarning)
        return left_dataset, right_dataset, Dataset()

    exclude_annotations_columns = list(exclude_annotations_columns)
    exclude_image_columns = list(exclude_image_columns)
    left_images, left_annotations = left_dataset.images, left_dataset.annotations
    right_images, right_annotations = right_dataset.images, right_dataset.annotations
    left_images = left_images.drop(columns=exclude_image_columns, errors="ignore")
    right_images = right_images.drop(columns=exclude_image_columns, errors="ignore")
    left_annotations = left_annotations.drop(
        columns=exclude_annotations_columns, errors="ignore"
    )
    right_annotations = right_annotations.drop(
        columns=exclude_annotations_columns, errors="ignore"
    )

    if exclude_annotations_columns or exclude_image_columns:
        # Construct new datasets with the excluded columns removed.
        # these one shoulb have the exact same columns
        return dataset_diff(
            left_dataset.loc[:, left_images.columns].loc_annot[
                :, left_annotations.columns
            ],
            right_dataset.loc[:, right_images.columns].loc_annot[
                :, right_annotations.columns
            ],
        )

    # If datasets are equal, just output left dataset with the excluded columns
    if left_images.equals(right_images) and left_annotations.equals(right_annotations):
        return left_dataset.loc[[]], right_dataset.loc[[]], left_dataset

    if set(left_images.columns) != set(right_images.columns):
        warn(
            "Column mismatch between Image DataFrames, consider using"
            " 'exclude_image_columns' argument, or comparing after manually excluding"
            " not shared columns in your datasets images frames",
            RuntimeWarning,
        )
        return left_dataset, right_dataset, Dataset()

    if set(left_annotations.columns) != set(right_annotations.columns):
        warn(
            "Column mismatch between Annotations DataFrames, consider using"
            " 'exclude_annotations_columns' argument, or comparing after manually"
            " excluding not shared columns in your datasets annotations frames",
            RuntimeWarning,
        )
        return left_dataset, right_dataset, Dataset()

    left_image_ids = set(left_images.index)
    right_image_ids = set(right_images.index)

    common_image_ids = left_image_ids & right_image_ids
    common_image_ids_list = list(common_image_ids)
    only_left_image_ids = left_image_ids - right_image_ids
    only_right_image_ids = right_image_ids - left_image_ids

    # compare images that share the same id across both datasets
    common_left_images = left_images.loc[common_image_ids_list]
    common_right_images = right_images.loc[common_image_ids_list]

    changed_values = common_left_images != common_right_images
    # None values should be considered the same here, the same pandas does with the
    # `equals` method
    changed_values[common_left_images.isna() & common_left_images.isna()] = False
    changed_images = changed_values.any(axis=1)
    changed_images_ids = set(common_left_images.loc[changed_images].index)

    left_annotations_ids = set(
        left_dataset.loc[common_image_ids_list].annotations.index
    )
    right_annotations_ids = set(
        right_dataset.loc[common_image_ids_list].annotations.index
    )
    changed_images_annotations_ids = set(
        left_dataset.loc[list(changed_images_ids)].annotations.index
    ) | set(right_dataset.loc[list(changed_images_ids)].annotations.index)

    # annotations that are the same but are linked to images that are different
    # are not considered the same
    common_annotations_ids = (
        left_annotations_ids & right_annotations_ids
    ) - changed_images_annotations_ids
    common_annotations_ids_list = list(common_annotations_ids)
    only_left_annotations_ids = left_annotations_ids - right_annotations_ids
    only_right_annotations_ids = right_annotations_ids - left_annotations_ids

    # Now compare annotations that share the same id and are linked to images that
    # are the same across datasets
    common_left_annot = left_annotations.loc[common_annotations_ids_list]
    common_right_annot = right_annotations.loc[common_annotations_ids_list]
    changed_annotations_values = common_left_annot != common_right_annot
    # Same as for images, None values in both frames should be considered the same
    changed_annotations_values[common_left_annot.isna() & common_right_annot.isna()] = (
        False
    )
    changed_annotations = changed_annotations_values.any(axis=1)
    changed_annotations_ids = set(common_left_annot.loc[changed_annotations].index)

    # resulting left dataset is composed of:
    # - images that are only in left dataset and all their annotations
    # - images that are different from right dataset and all
    #   their annotations
    # - annotation that are only in left dataset and their linked image
    # - annotations that are different from right dataset and their linked image
    only_left_dataset = (
        left_dataset.loc[list(only_left_image_ids | changed_images_ids)]
        + left_dataset.loc_annot[
            list(only_left_annotations_ids | changed_annotations_ids)
        ].remove_empty_images()
    )

    # Same applied for right dataset
    only_right_dataset = (
        right_dataset.loc[list(only_right_image_ids | changed_images_ids)]
        + right_dataset.loc_annot[
            list(only_right_annotations_ids | changed_annotations_ids)
        ].remove_empty_images()
    )

    # Resulting common dataset is composed of images that are in both datasets and
    # that are the same across datasets, and only their annotations if they are
    # in both datasets and are the same across datasets.
    common_dataset = (
        left_dataset.loc[list(common_image_ids - changed_images_ids)]
        .loc_annot[list(common_annotations_ids - changed_annotations_ids)]
        .remove_empty_images()
    )

    return only_left_dataset, only_right_dataset, common_dataset
