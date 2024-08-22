import warnings
from pathlib import Path

import pandas as pd
from typing_extensions import Self

from ..dataset import Dataset
from ..utils.label_map_merger import merge_label_maps
from ..utils.parquet_saver import dict_from_parquet, dict_to_parquet
from ..utils.testing import assert_frame_intersections_equal


class Evaluator:
    """Abstract class of Evaluator, made to measure prediction quality with respect to a
    Dataset of groundtruth annotations. Depending on data type, the method used for
    evaluation might differ, refer to the specialized classes for information. The
    fundamental building block is the Dataset object representing the groundtruth.
    additional kwargs given to the constructor are also Dataset objects that must match
    the groundtruth, in terms of image and label maps (if any)
    """

    name: str | None
    """Name of Evaluator. Can be deduced from groundtruth's dataset name and will be
    used in export functions like :meth:`.DetectionEvaluator.to_fiftyone`"""

    groundtruth: pd.DataFrame
    """DataFrame comprising annotation data. Must have at least ``image_id`` column"""

    predictions_dictionary: dict[str, pd.DataFrame]
    """dictionary of DataFrames comprising prediction data. Must have at least
    ``image_id`` and ``confidence`` columns"""

    images: pd.DataFrame
    """DataFrame comprising image data. This dataframe should be referred to by both gt
    and predictions with the ``image_id`` column"""

    images_root: Path
    """Root folder where to grab images. Image filepath will be concatenation of
    images_root and their relative path"""

    label_map: dict[int, str]
    """Mapping from category_id to category_str. If used, is generally taken from the
    groundtruth Dataset. The prediction must be compatible with it"""

    def __init__(
        self,
        groundtruth: Dataset,
        name: str | None = None,
        **predictions: Dataset,
    ):
        """Constructor of the Evaluator object.

        Args:
            groundtruth: Dataset object representing the ground truth with
                annotations, image data and label_map
            name: Name of Evaluator. If set to None, will be deduced from groundtruth's
                dataset name
            **predictions: keyword arguments for additional datasets to compare the
                groundtruth to. Its images must match the groundtruth dataset (see
                add_prediction_dataset method below).
        """
        if name is None:
            self.name = groundtruth.dataset_name
        else:
            self.name = name
        self.images_root = groundtruth.images_root
        self.groundtruth = groundtruth.annotations
        self.images = groundtruth.images.drop("split", axis=1, errors="ignore")
        self.label_map = groundtruth.label_map
        self._default_annotation_columns_with_types = (
            groundtruth._default_annotation_columns_with_types
        )
        self._default_image_columns_width_types = (
            groundtruth._default_image_columns_with_types
        )
        self.predictions_dictionary = {}
        for predictions_name, predictions_df in predictions.items():
            self.add_predictions_dataset(predictions_name, predictions_df)

    def get_image_attributes(self) -> list[str]:
        """Get the name of columns related to image attributes. In other words, get
        columns that are NOT the default ones.

        The actual attribute values can then be
        ``self.images[self.get_image_attributes()]``

        Returns:
            list of column names in ``self.images`` that represent attributes
        """
        return [
            str(c)
            for c in self.images.columns
            if c not in self._default_image_columns_width_types.keys()
        ]

    def get_annotations_attributes(
        self, predictions_name: str | None = None
    ) -> list[str]:
        """Get the name of columns related to annotations attributes. In other words,
        get columns that are NOT the default ones.

        the actual attribute values can then be

        .. code-block:: python

            self.predictions_dictionary[predictions_name][
                self.get_annotations_attributes()
            ]

        Args:
            predictions_name: name of predictions to extract not default column from.
                If None, will use ``self.groundtruth``. Defaults to None.

        Returns:
            list of column names in ``self.annotations`` that represent attributes
        """
        if predictions_name is None:
            predictions = self.groundtruth
        else:
            predictions = self.predictions_dictionary[predictions_name]
        return [
            str(c)
            for c in predictions.columns
            if c not in self._default_annotation_columns_with_types.keys()
        ]

    def add_predictions_dataset(self, predictions_name: str, predictions: Dataset):
        """Method to add predictions to the Evaluator from a Dataset object.
        The prediction dataset must match the Evaluator data:

        - prediction label_map must be equal or a subset to the
            Evaluator's label map
        - image data must be the same, except the relative path
            (it can change although the image has not) i.e. there must be
            the same number and ids of images and all the columns in the prediction
            image data must match the corresponding ones
            in the evaluator image data.

        Note that this method will overwrite a potentially already existing prediction
        dataframe

        Args:
            predictions_name: name of predictions to add. It will then be used as key in
                the ``self.predictions_dictionary`` attribute.
            predictions: prediction Dataset, from which the annotations will
                be extracted and added to the evaluator.
        """
        assert "confidence" in predictions.annotations, "Not a prediction dataset"

        new_label_map = merge_label_maps(
            self.label_map, predictions.label_map, method="outer"
        )
        if new_label_map != self.label_map:
            warnings.warn(
                f"Although compatible, '{predictions_name}' prediction label map is"
                " larger than groundtruth label map",
                RuntimeWarning,
            )
            self.label_map = new_label_map

        if not predictions.images.index.isin(self.images.index).all():
            raise ValueError(
                "Some image ids in given predictions are not present in the evaluator"
                " image index"
            )

        try:
            assert_frame_intersections_equal(self.images, predictions.images)
        except AssertionError as e:
            raise AssertionError(
                "Groundtruth and Prediction images are not consistent on their"
                " overlapping indices and columns. You might want to consider the"
                " Dataset.reindex() method."
            ) from e

        self.add_predictions(predictions_name, predictions.annotations)

    def add_predictions(self, predictions_name: str, predictions: pd.DataFrame):
        """Method to add predictions to the Evaluator from a dataframe.
        No check will be done on image data the annotations refer to. However, it will
        check that ``image_id`` values of ``predictions`` are contained in the
        evaluator's ``image_data`` and ``category_id`` values are contained in the
        label map

        Note that this method will overwrite a potentially already existing prediction
        dataframe

        Args:
            predictions_name: name of predictions to add. It will then be used as key in
                the ``self.predictions_dictionary`` attribute.
            predictions: prediction dataframe to be added to
                the evaluator.
        """
        predictions_image_ids = set(predictions["image_id"].unique())
        assert set(self.images.index).issuperset(predictions_image_ids)
        predictions_class_ids = set(predictions["category_id"].unique())
        assert predictions_class_ids.issubset(self.label_map.keys())
        self.predictions_dictionary[predictions_name] = predictions

    def to_parquet(self, output_dir: Path | str, overwrite: bool = False) -> None:
        """Save the current object to a folder containing parquet files for dataframes
        inside this object, and a metadata.yaml file for other attributes.

        Args:
            output_dir: output directory where the files will be created.
                If ``overwrite`` is set to False, it must not already exist.
            overwrite: if set to True, will remove the directory at ``output_dir``
                if it already exists. Defaults to False.
        """
        dict_to_parquet(
            {k: v for k, v in vars(self).items() if not k.startswith("_")}
            | {"__name__": self.__class__.__name__},
            Path(output_dir),
            overwrite=overwrite,
        )

    @classmethod
    def from_parquet(cls, input_dir: Path | str) -> Self:
        """Class method to construct an instance of this class or a subclass.
        the parquet folder must have been created with the method ``to_parquet``
        (see above)

        Args:
            input_dir: Path to directory containing the metadata.yaml file along with
                the different parquet files

        Raises:
            ValueError: Raised when the object name contained in
                ``input_dict['__name__']`` is not the same as the name of the class this
                method is called from. For example, you can't call
                :meth:`.Evaluator.from_parquet` with a folder created by a
                :class:`DetectionEvaluator` object.

        Returns:
            New object of the same subclass as the method is caled from,
            containing data loaded from the parquet files in the input directory
        """
        input_dict = dict_from_parquet(Path(input_dir))
        if cls.__name__ != input_dict["__name__"]:
            raise ValueError(
                f"Wrong object type for parquet archive. Expected {cls.__name__}, got"
                f" {input_dict['__name__']}"
            )
        groundtruth_dataset = Dataset(
            images_root=input_dict["images_root"],
            images=input_dict["images"].assign(split=None),
            annotations=input_dict["groundtruth"],
            label_map=input_dict["label_map"],
        )
        predictions = input_dict["predictions_dictionary"]
        evaluator = cls(groundtruth_dataset)
        for name, predictions in predictions.items():
            evaluator.add_predictions(name, predictions)
        for k, v in vars(evaluator).items():
            loaded_value = input_dict.get(k, None)
            if loaded_value is not None:
                evaluator.__dict__[k] = loaded_value
        return evaluator

    def _ipython_display_(self):
        """Function to display the Dataset as an HTML widget when using notebooks"""
        import ipywidgets as widgets
        from IPython.display import display

        tab = widgets.Tab()

        descr_str = (
            "<b> Evaluation object, containing "
            f"{len(self.images):,} images, {len(self.groundtruth):,} groundtruth "
            f"objects, and {len(self.predictions_dictionary)} prediction sets </b>"
        )

        title = widgets.HTML(descr_str)

        components_widgets = self._get_widgets()

        tab.children = [*components_widgets.values()]
        tab.titles = [*components_widgets.keys()]

        display(widgets.VBox([title, tab]))

    def _get_widgets(self):
        import ipywidgets as widgets
        from IPython.display import display

        label_map_df = pd.Series(self.label_map, name="category string").to_frame()
        label_map_df.index.name = "categorty_id"

        # create output widgets
        widget_images = widgets.Output()
        widget_groundtruth = widgets.Output()
        widget_predictions = {
            p_name: widgets.Output() for p_name in self.predictions_dictionary
        }
        widget_label_map = widgets.Output()

        # render in output widgets
        with widget_images:
            display(self.images)
        with widget_groundtruth:
            display(self.groundtruth)
        for p_name, p in self.predictions_dictionary.items():
            with widget_predictions[p_name]:
                display(p)
        with widget_label_map:
            display(label_map_df)

        return {
            "Images": widget_images,
            "Groundtruth": widget_groundtruth,
            **widget_predictions,
            "label_map": widget_label_map,
        }
