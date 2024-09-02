"""Set of functions used to test some assertions on datasets.
Useful when used in unit tests
"""

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
from imageio.v3 import imread
from pandas.testing import assert_frame_equal
from tqdm.auto import tqdm

from ..dataset import Dataset
from ..utils import BBOX_COLUMN_NAMES

BOX_XMIN, BOX_YMIN, BOX_WIDTH, BOX_HEIGHT = BBOX_COLUMN_NAMES


def assert_column(
    input_df: pd.DataFrame,
    assertion: pd.Series | np.ndarray,
    message: str = "",
    n_first_occurrences: int | None = 1,
) -> None:
    """From a given input dataframe and a boolean series of the same length, construct
    an error message if the boolean has at least one False value, with the row in
    input dataframe corresponding to the row of the first occurrence of False value in
    the assertion series

    Args:
        input_df: Dataframe to show the row from, to better understand what went wrong
        assertion: Boolean Series of the same length as ``input_df``, expected to be
            full of True value
        message: Message to display when raising the error. Will be followed with
            information of faulty rows
        n_first_occurrences: Number of occurrences to show in case of a failure. Useful
            when showing duplicate values. If set to None, will show all occurrences.

    Raises:
        AssertionError: If there is at least one occurrence of False in ``assertion``
            Series, raise an assertion and print the corresponding row of first
            occurrence in ``input_df``
    """
    assert len(input_df) == len(assertion)
    assert n_first_occurrences is None or n_first_occurrences > 0
    assertion = assertion.astype(bool)
    if not assertion.all():
        failure = input_df[~assertion].iloc[:n_first_occurrences]
        if n_first_occurrences is None:
            raise AssertionError(f"Assertion failed. {message}. {failure}")
        elif n_first_occurrences == 1:
            raise AssertionError(
                f"Assertion failed. {message}. First occurrence at row"
                f" {failure.index[0]} : {failure.iloc[0]}"
            )
        else:
            raise AssertionError(
                f"Assertion failed. {message}. First occurrences at rows"
                f" {failure.index[:n_first_occurrences]} :\n{failure.iloc[:n_first_occurrences]}"
            )


def assert_columns_properly_normalized(
    input_df: pd.DataFrame, separator: str = "."
) -> None:
    """Checks that columns in input dataframes are well normalized, i.e. checks that
    if column 'A' exists, column 'A.B' does not exists.

    This is useful when loading json files to checks that a key cannot be both a
    sub dictionary and a value

    Args:
        input_df: Input DataFrame to test
        separator: Character used to separate name in flattened key. Defaults to ".".

    Raises:
        AssertionError: if there exist a column name where both the name and a variation
            of name + separator exists
    """
    for c in input_df.columns:
        prefix = f"{c}{separator}"
        for c2 in input_df.columns:
            if c2.startswith(prefix):
                raise AssertionError(
                    f"DataFrame is not properly normalized. Column '{c}' cannot be"
                    f" both a value and a subdictionary, but column '{c2}' exists"
                )


def assert_dataset_equal(
    dataset1: Dataset,
    dataset2: Dataset,
    ignore_index: bool = False,
    optional_columns: Iterable[str] = ("area", "confidence"),
    remove_na_columns: bool = False,
) -> None:
    """Compare two datasets and raise an assertion error if datasets are not equal.
    This function is mainly intended to be used in the context of unit tests.

    Rules:
        - Index order is not relevant. This is similar to ``check_like`` option
          in :func:`pandas.testing.assert_frame_equal`
        - Indexes for rows and columns still must be the same when reordered
        - Some columns in annotations are optional and are thus ignored if present in
          one but not the other dataset.
          If both are present, the columns' values are still compared.
        - Label maps must be the same. Again, order is ignored (as it normally is for
          dictionaries)
        - If ``ignore_index`` option is set to ``True``, index for rows are not checked,
          but we still check that the key in annotations' ``image_id`` points to the
          same rows in images dataframe

    Args:
        dataset1: First dataset to test
        dataset2: Second dataset to test, must be the same according to mentioned rules
            or the function will raise an error
        ignore_index: If set, will ignore both annotations and images dataframe index,
            but will still check that link between annotations and image row with
            ``image_id`` is the same. Defaults to False.
        optional_columns: Iterable of column names that will considered as optional,
            i.e. only check them if they are both present. Defaults to the column names
            "area" and "confidence".
        remove_na_columns: If set to True, will remove from dataframes columns where all
            values are equivalent to panda's ``<NA>``. This more lenient comparison is
            useful for columns where its absence and its values being all ``<NA>`` are
            treated the same, like the ``split`` column.

    Raises:
        AssertionError: raised when datasets are detected to be different
    """
    optional_columns = list(optional_columns)

    def assert_frame_equal_optional_columns(
        frame1: pd.DataFrame,
        frame2: pd.DataFrame,
        optional_columns: Sequence[str],
        dataframe_name: str,
    ) -> None:
        """Assert dataframe are equal, but first remove the optional columns if they are
        present in one dataframe and not in the other. Otherwise, if present in both,
        keep them for comparison
        """
        if remove_na_columns:
            frame1 = frame1.dropna(axis="columns", how="all")
            frame2 = frame2.dropna(axis="columns", how="all")
        for column_name in optional_columns:
            if column_name in frame1.columns and column_name not in frame2.columns:
                frame1 = frame1.drop(column_name, axis=1)
            if column_name in frame2.columns and column_name not in frame1.columns:
                frame2 = frame2.drop(column_name, axis=1)
        try:
            assert_frame_equal(frame1, frame2, check_like=True, check_dtype="equiv")
        except AssertionError as e:
            raise AssertionError(f"{dataframe_name} dataframes don't match") from e

    if ignore_index:
        dataset1 = dataset1.reset_index()
        dataset2 = dataset2.reset_index()

    assert_frame_equal_optional_columns(
        dataset1.images, dataset2.images, optional_columns, "Images"
    )
    assert_frame_equal_optional_columns(
        dataset1.annotations, dataset2.annotations, optional_columns, "Annotations"
    )
    assert (
        dataset1.label_map == dataset2.label_map
    ), f"label_maps don't match {dataset1.label_map} vs {dataset2.label_map}"

    assert dataset1.booleanized_columns == dataset2.booleanized_columns


def assert_frame_intersections_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Construct inner dataframes from overlapping ids and columns and check they are
    equal

    These are the rows and columns present in both images dataframes
    The two dataframes must have the same values for the merge to be valid

    Args:
        df1: First dataframe to test
        df2: Second dataframe to test

    Raises:
        AssertionError: Raise error if both subdataframe constructed with intersections
            of indexes and columns are not the same.
    """
    df1_ids = set(df1.index)
    df2_ids = set(df2.index)
    mutual_ids = list(df1_ids & df2_ids)

    if not mutual_ids:
        return

    df1_columns = set(df1.columns)
    df2_columns = set(df2.columns)
    mutual_columns = list(df1_columns & df2_columns)

    if not mutual_columns:
        return

    inner_df1 = df1.loc[mutual_ids, mutual_columns]
    inner_df2 = df2.loc[mutual_ids, mutual_columns]
    try:
        assert_frame_equal(inner_df1, inner_df2)
    except AssertionError as e:
        raise AssertionError(
            "sub-Dataframes constructed from ids and columns in both DataFrames are not"
            " equal."
        ) from e


def assert_images_valid(
    dataset: Dataset,
    assert_is_symlink: bool = False,
    load_images: bool = True,
    check_exhaustive: bool = False,
) -> None:
    """Checks that the image paths in the dataset. Namely, checks that all path
    are indeed pointing to a file, and are valid file format that can be loaded with
    ``imageio``.

    Note:
        Todo: better error messages

    Args:
        dataset: Dataset to check
        assert_is_symlink: If set, will check that paths are symlinks rather than
            files. Defaults to False.
        load_images: If set to True, will not only check that images are valid files,
            but also that image can be loaded (i.e. are not corrupted files) and that
            their sizes match the ones included in ``dataset.images``
            dataframe. Note that this makes the function significantly slower.
            Defaults to True.
        check_exhaustive: If set to True, will check that all images in the images_root
            folder are in the image dataframe, and that the dataset is indeed exhaustive
    """
    get_invalid_images(
        dataset, assert_is_symlink, load_images, check_exhaustive, raise_if_error=True
    )


class InvalidImage(AssertionError):
    pass


class MissingImages(AssertionError):
    pass


def get_invalid_images(
    dataset: Dataset,
    check_symlink: bool = False,
    load_images: bool = True,
    check_exhaustive: bool = False,
    raise_if_error: bool = True,
) -> pd.DataFrame:
    """Checks dataset's images and return an indexed error report to retrieve them.

    Namely, checks that all path are indeed pointing to a file, and are valid file
    format that can be loaded with ``imageio``. If unsuccessful, add a row to the output
    dataframe with the same index as the faulty images, and info about the error in
    corresponding columns

    Args:
        dataset: Dataset to check
        check_symlink: If set, will check that paths are symlinks rather than
            files. Defaults to False.
        load_images: If set to True, will not only check that images are valid files,
            but also that image can be loaded (i.e. are not corrupted files) and that
            their sizes match the ones included in ``dataset.images``
            dataframe. Note that this makes the function significantly slower.
            Defaults to True.
        check_exhaustive: If set to True, will check that all images in the images_root
            folder are in the image dataframe, and that the dataset is indeed exhaustive
        raise_if_error: If set to True, will raise an InvalidImage error as soon as
            one image does not meet the requirements.

    Raises:
        InvalidImage: Raised if ``raise_if_error`` is selected and one image is not
            valid. Can be because the path is not right, the image loading failed,
            or the metadata is not compliant with actual image data.
        MissingImages: Raised if ``raise_if_error`` is selected and some images
            where found in the ``images_root`` folder but not in the dataset's
            ``images`` dataframe.

    Returns:
        Error report in the form of a Dataframe with "reason" and "additional_info"
        columns. Index values are the same as the corresponding images in the original
        dataset, so that you can retrieve the faulty images full data.
    """
    error_report = {}

    def error(
        message: str,
        additional_info: str,
        row_id: Hashable,
        image_data: "pd.Series[str]",
    ) -> None:
        error_report[row_id] = {"reason": message, "additional_info": additional_info}
        if raise_if_error:
            raise InvalidImage(
                f"{message}, {additional_info}\n row : {row_id}\n data:"
                f" {image_data.to_dict()}"
            )

    for row, img_data in tqdm(dataset.images.iterrows(), total=len(dataset)):
        if img_data["relative_path"].is_absolute():
            error("relative path is absolute", "", row, img_data)
            continue
        img_path = dataset.images_root / img_data["relative_path"]
        if check_symlink and not img_path.is_symlink():
            error("Not a symlink", "", row, img_data)
            continue
        valid_path = (
            img_path.is_symlink()
            and img_path.readlink().is_file()
            or img_path.is_file()
        )
        if not valid_path:
            error("Not a valid path", "", row, img_data)
            continue
        if load_images:
            try:
                img = imread(img_path)
            except OSError:
                error(
                    "corrupted file",
                    "Image cannot be loaded with imageio",
                    row,
                    img_data,
                )
                continue
            if not isinstance(img_data["width"], int) or img_data["width"] <= 0:
                error(
                    "Invalid image width",
                    f"got {img_data['width']} pixels",
                    row,
                    img_data,
                )
                continue
            if not isinstance(img_data["height"], int) or img_data["height"] <= 0:
                error(
                    "Invalid image height",
                    f"got {img_data['height']} pixels",
                    row,
                    img_data,
                )
                continue
            if len(img.shape) not in [2, 3, 4]:
                error(
                    "invalid image shape",
                    f"Shape is with {len(img.shape)} dimensions instead of 2"
                    " (grayscale), 3 (RGB/RGBA) or 4 (GIf anim)",
                    row,
                    img_data,
                )
            if len(img.shape) == 4:
                _, height, width, _ = img.shape
            elif len(img.shape) == 3:
                height, width, _ = img.shape
            else:
                height, width = img.shape
            if img_data["width"] != width:
                error(
                    "Image width in metadata is different from actual image width",
                    f"{width} (actual) vs {img_data['width']} (metadata)",
                    row,
                    img_data,
                )
            if img_data["height"] != height:
                error(
                    "Image height in metadata is different from actual image height",
                    f"{height} (actual) vs {img_data['height']} (metadata)",
                    row,
                    img_data,
                )
    if check_exhaustive:
        from ..dataset import from_folder

        highest_id = dataset.images.index.max()
        all_images = (
            from_folder(images_root=dataset.images_root)
            .reset_index(start_image_id=highest_id + 1)
            .images
        )
        missing_images = all_images.loc[
            ~all_images["relative_path"].isin(dataset.images["relative_path"]),
            "relative_path",
        ].apply(str)
        if len(missing_images) > 0 and raise_if_error:
            raise AssertionError(
                "Dataset is not exhaustive : the following images are present in"
                " images root but not in dataset image dataframe"
                f" :\n{', '.join(missing_images)}"
            )
        for row, relative_path in missing_images.items():
            error_report[row] = {
                "message": "missing image",
                "additional_info": relative_path,
            }

    return pd.DataFrame.from_dict(error_report, orient="index")


def assert_ids_well_formed(dataset: Dataset) -> None:
    """Assert ids follow the right convention.

    - DataFrames indexes must be named "id"
    - indexes must have no duplicates
    - images ``relative_path`` column must have no duplicates
    - annotation ``image_id`` values must all be in images index
    - annotation ``category_id`` values must be in dataset's label map

    Note:
        Todo: Better error messages

    Args:
        dataset: Dataset object to test.
    """
    assert dataset.images.index.name == "id", (
        "dataset's image index must be named 'id', got"
        f" {dataset.images.index.name} instead"
    )

    assert_column(
        dataset.images,
        ~dataset.images.index.duplicated(keep=False),
        "Dataset image index has duplicate values",
    )

    assert_column(
        dataset.images,
        ~dataset.images["relative_path"].duplicated(keep=False),
        "Dataset image relative path has duplicate values",
    )
    assert dataset.annotations.index.name == "id", (
        "dataset's annotation index must be named 'id', got"
        f" {dataset.annotations.index.name} instead"
    )
    assert_column(
        dataset.annotations,
        ~dataset.annotations.index.duplicated(keep=False),
        "Dataset annotations index has duplicate values",
    )
    assert_column(
        dataset.annotations,
        dataset.annotations["image_id"].isin(dataset.images.index),
        "All image_id values in annotations must in dataset images index",
    )
    assert_column(
        dataset.annotations,
        dataset.annotations["category_id"].isin(dataset.label_map.keys()),
        "All category ids must be in dataset label map",
    )


def assert_bounding_boxes_well_formed(
    dataset: Dataset, allow_keypoints: bool = False
) -> None:
    """Assert bounding boxes are well-formed in dataset's annotations.

    - Boxes x and y coordinates must be within their respective image size
    - Boxes width and height must be positive and so that xmax and ymin are within
      their respective image size
    - in the case of keypoints, Boxes with size 0 will be tolerated

    Args:
        dataset: Dataset to test
        allow_keypoints: If set to True, will not raise error if
            bounding box size (width or height) is 0. Defaults to False.
    """
    get_malformed_bounding_boxes(dataset, allow_keypoints, raise_if_error=True)


def get_malformed_bounding_boxes(
    dataset: Dataset, allow_keypoints: bool = False, raise_if_error: bool = False
) -> pd.DataFrame:
    """Get malformed bounding in dataset's annotations, as a boolean dataframe where
    index is id of bounding box in dataset's annotations dataframe, and columns are
    known reasons for bounding boxes to be invalid

    - Boxes x and y coordinates must be within their respective image size
    - Boxes width and height must be positive and so that xmax and ymin are within
      their respective image size
    - in the case of keypoints, Boxes with size 0 will be tolerated

    An invalid bounding box is then related to a row in the result dataframe where at
    least one of the value is True. Note that valid bounding boxes are NOT in the result
    dataframe. This means that if the dataset has no invalid bounding box, the result
    dataframe will be empty, and for each row in the result dataframe, there will be at
    least one ``True`` value.

    Args:
        dataset: Dataset to test
        allow_keypoints: If set to True, will not raise error if
            bounding box size (width or height) is 0. Defaults to False.
        raise_if_error: If set to True, will raise an error as soon as one bounding box
            is detected to be invalid. Defaults to False.

    Raises:
        AssertionError: When ``raise_if_error`` is set, raise an error as soon as one
            bounding box is invalid.

    Returns:
        Error report as a dataframe with boolean columns.

        - Each column is a reason why the bounding box can be faulty.
        - Each row is a faulty bounding box, with its corresponding index in dataset's
          annotation dataframe. Its value explain how the bounding box is invalid.
        - Only the faulty bounding boxes are kept in the error report, so all rows have
          at least one value set to True.
    """
    error_report = pd.DataFrame(index=dataset.annotations.index)

    def report_if_error(
        assertion: "pd.Series[bool]",
        error_name: str,
        message: str,
    ) -> None:
        error_report[error_name] = ~assertion
        if raise_if_error:
            assert_column(
                dataset.annotations, assertion, message, n_first_occurrences=1
            )

    report_if_error(
        dataset.annotations[BOX_XMIN] >= 0,
        "Negative X value",
        "Bounding boxes must have positive X values",
    )

    report_if_error(
        dataset.annotations[BOX_YMIN] >= 0,
        "Negative Y value",
        "Bounding boxes must have positive Y values",
    )
    report_if_error(
        dataset.annotations[BOX_WIDTH] >= 0,
        "Negative width",
        "Bounding boxes must have positive width",
    )
    report_if_error(
        dataset.annotations[BOX_HEIGHT] >= 0,
        "Negative height",
        "Bounding boxes must have positive height",
    )
    if not allow_keypoints:
        report_if_error(
            dataset.annotations[BOX_WIDTH] > 0,
            "0 width",
            "Bounding boxes must have strictly positive width",
        )
        report_if_error(
            dataset.annotations[BOX_HEIGHT] > 0,
            "0 height",
            "Bounding boxes must have strictly positive height",
        )
    x_max = dataset.annotations[BOX_XMIN] + dataset.annotations[BOX_WIDTH]
    y_max = dataset.annotations[BOX_YMIN] + dataset.annotations[BOX_HEIGHT]
    im_width = dataset.images.loc[dataset.annotations["image_id"], "width"]
    im_height = dataset.images.loc[dataset.annotations["image_id"], "height"]
    im_width.index = x_max.index
    im_height.index = x_max.index

    report_if_error(
        x_max <= im_width,
        "right side outside of image",
        "Bounding boxes must have X values below image width",
    )
    report_if_error(
        y_max <= im_height,
        "bottom side outside of image",
        "Bounding boxes must have Y values below image height",
    )
    return error_report[error_report.any(axis=1)]


def assert_label_map_well_formed(dataset: Dataset) -> None:
    """Assert label map has no category name duplicate

    Args:
        dataset: dataset to test.
    """
    label_map = pd.Series(dataset.label_map)
    assert (
        not label_map.duplicated().any()
    ), f"Label map dictionary has duplicate values : {dataset.label_map}"


def assert_required_columns_present(
    input_df: pd.DataFrame, required_columns: set[str], df_name: str
) -> None:
    """Simple function to check that required columns are present and raise a custom
    error if it's not the case

    Args:
        input_df: dataframe object to check.
        required_columns: set of column names to find in the columns of ``input_df``.
        df_name: name of the dataframe, used to add context to the error message.

    Raises:
        ValueError: Raised when not all required columns are present in the
            columns of ``input_df``.
    """
    missing_columns = required_columns - set(input_df.columns)
    if missing_columns:
        raise ValueError(
            f"DataFrame {df_name} must have all these columns"
            f" :\n{', '.join(required_columns)}\nbut is missing"
            f" {', '.join(missing_columns)}"
        )


def full_check_dataset_detection(
    dataset: Dataset,
    check_symlink: bool = False,
    allow_keypoints: bool = False,
    check_exhaustive: bool = False,
) -> None:
    """Perform a full check of the dataset. Images must be reachable for the test to
    perform.

    Args:
        dataset: dataset to test
        check_symlink: If set to True, will check that image relative paths are indeed
            relative links and not actual files. Defaults to False.
        allow_keypoints: If set to True, will not raise an error for bounding boxes with
            size 0 (width or height). Defaults to False.
        check_exhaustive: If set to True, will check that all images in the images_root
            folder are in the image dataframe, and that the dataset is indeed exhaustive
    """
    print("Checking Image and annotations Ids ...")
    assert_ids_well_formed(dataset)
    print("Checking Bounding boxes ..")
    assert_bounding_boxes_well_formed(dataset, allow_keypoints=allow_keypoints)
    print("Checking label map ...")
    assert_label_map_well_formed(dataset)
    print("Checking images are valid ...")
    get_invalid_images(
        dataset, check_symlink, True, check_exhaustive, raise_if_error=True
    )
