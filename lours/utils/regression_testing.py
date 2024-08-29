import os
from hashlib import sha256
from pathlib import Path

import pytest
from imageio import imread
from pytest_regressions.common import perform_regression_check

from lours.dataset import Dataset, from_parquet

from .testing import assert_dataset_equal


class DataSetRegressionFixture:
    """Lours Dataset Regression fixture implementation used on dataset_regression
    fixture.

    Loosely based on pytest-regression's DataFrameRegression.
    See `Related code on github`__

        .. __: https://github.com/ESSS/pytest-regressions/blob/master/src/pytest_regressions/dataframe_regression.py
    """

    def __init__(
        self, datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
    ) -> None:
        self.request = request
        self.datadir = datadir
        self.original_datadir = original_datadir
        self._force_regen = False
        self._with_test_class_names = False

    def check_fn(
        self, obtained_dataset_path: Path, expected_dataset_path: Path
    ) -> None:
        """Check that two saved parquet datasets are the same

        Args:
            obtained_dataset_path: path to current dumped dataset
            expected_dataset_path: path to saved dataset to compare the current
                object with.
        """
        actual_obtained_path = obtained_dataset_path.parent / (
            obtained_dataset_path.name + ".d"
        )
        actual_expected_path = expected_dataset_path.parent / (
            expected_dataset_path.name + ".d"
        )
        dataset1 = from_parquet(actual_obtained_path)
        dataset2 = from_parquet(actual_expected_path)

        assert_dataset_equal(dataset1, dataset2)

    def dump_fn(
        self, dataset: Dataset, filename: Path, with_image_checksum: bool = False
    ) -> None:
        """Function used to dump the dataset.

        Note:
            Since pytest-regressions only accepts files and :func:`.to_parquet()`
            method creates a folder, the dump function here will create an empty file
            and dump everything in a folder named after the file path with the ".d"
            added to it.

        Args:
            dataset: dataset to save
            filename: path to file that will be considered the file of the dataset.
                the actual dataset will be saved in the same path but with ".d" appended
                to it.
            with_image_checksum: If set to True, will load all images to get a checksum
                for each image based on its pixels, and save it in a dedicated column in
                the dataset's image dataframe.
        """
        output_folder = filename.parent / (filename.name + ".d")
        if with_image_checksum:
            checksums = {}
            for image_id, relative_path in dataset.images["relative_path"].items():
                image_path = dataset.images_root / relative_path
                image = imread(image_path)
                checksums[image_id] = sha256(image.tobytes()).hexdigest()
            dataset = dataset.from_template(
                images=dataset.images.assign(checksum=checksums)
            )
        dataset.to_parquet(output_folder, overwrite=True)
        # If dump was successful, make an empty file so that filename is a path to a
        # valid file, making pytest-regression happy
        filename.touch()

    def check(
        self,
        dataset: Dataset,
        basename: str | None = None,
        fullpath: "os.PathLike[str] | None" = None,
        check_images: bool = False,
    ) -> None:
        """Checks a dataset object, against a previously recorded version, or generate a
        new parquet version of it.

        Note:
            Since pytest-regressions only accepts files and :func:`.to_parquet()`
            method creates a folder, the dump function here will create an empty file
            and dump everything in a folder named after the file path with the ".d"
            added to it.


        Args:
            dataset: Dataset containing annotations for regression check
            basename: basename of the file to test/record. If not given the name
                of the test is used.
            fullpath: complete path to use as a reference file. This option
                will ignore ``embed_data`` completely, being useful if a reference
                file is located in the session data dir for example. Defaults to None.
            check_images: If set to True, will load all images to get a checksum of each
                image, based on its pixels, and save it in a dedicated column in
                the dataset's image dataframe. Defaults to False
        """
        import functools

        assert isinstance(dataset, Dataset), (
            "Only Lours datasets are supported on dataframe_regression fixture.\n"
            f"Object with type {type(dataset)} was given."
        )

        dump_fn = functools.partial(
            self.dump_fn, dataset, with_image_checksum=check_images
        )

        perform_regression_check(
            datadir=self.datadir,
            original_datadir=self.original_datadir,
            request=self.request,
            check_fn=self.check_fn,
            dump_fn=dump_fn,
            extension="",
            basename=basename,
            fullpath=fullpath,
            force_regen=self._force_regen,
            with_test_class_names=self._with_test_class_names,
        )


@pytest.fixture
def dataset_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "DataSetRegressionFixture":
    """Regression fixture for datasets. Will save the dataset to parquet and compare
    with the already saved parquet dataset

    Args:
        datadir: Where to save the dataset in parquet format. If not set, will use the
            default value from pytest-data, i.e. folder with the name of the python
            file + file with the name of the test function.
        original_datadir: Where to load the original dataset, in parquet format. If not
            set, will use the default value from pytest-data, i.e. folder with the name
            of the python file + file with the name of the test function.
        request: pytest request object used by pytest-regression to get options from
            pytest shell command.

    Returns:
        Fixture used by pytest to dump a dataset, and compare its dump with the
        already existing one.

    Example:

        >>> from lours.utils.doc_utils import dummy_dataset
        >>> def testSomeData(dataset_regression):
        ...     my_dataset = dummy_dataset()
        ...     dataset_regression.check(my_dataset)
        ...
    """
    return DataSetRegressionFixture(datadir, original_datadir, request)
