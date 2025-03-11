from collections.abc import Iterable, Iterator, Sequence
from copy import deepcopy
from os.path import normpath, relpath
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Literal

try:
    from typing import Self
except ImportError:
    # Fallback mechanism for python 3.10
    from typing_extensions import Self
from warnings import warn

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas._typing import Dtype

from ..utils import BBOX_COLUMN_NAMES
from ..utils.column_booleanizer import booleanize, debooleanize, get_bool_columns
from ..utils.grouper import get_group_names, group_list, groups_to_list
from ..utils.label_map_merger import IncompatibleLabelMapsError
from ..utils.parquet_saver import dict_to_parquet
from .split.dataset_splitter import split_dataframe

if TYPE_CHECKING:
    import fiftyone as fo

    from ..utils.annotations_appender import AnnotationAppender
    from .indexing import DatasetAnnotLocator, DatasetImLocator


class Dataset:
    """Dataset base class for manipulation

    The behaviour of the Dataset is inspired from numpy arrays or pandas dataframes.

    See Also:
        - `related doc <UPDATE-ME>`_
          for a complete explanation of main principles.
        - :ref:`Dataset demo notebook </notebooks/1_demo_dataset.ipynb>`

    """  # noqa: E501

    dataset_name: str | None
    images_root: Path
    images: pd.DataFrame
    annotations: pd.DataFrame
    label_map: dict[int, str]
    _image_required_columns: set[str] = {"width", "height", "relative_path"}
    _default_image_columns_with_types: dict[str, Dtype] = {
        "width": int,
        "height": int,
        "relative_path": object,
        "type": str,
        "split": str,
    }
    _annotations_required_columns: set[str] = {
        "image_id",
        "category_id",
        *BBOX_COLUMN_NAMES,
    }
    _default_annotation_columns_with_types: dict[str, Dtype] = {
        "image_id": int,
        "category_str": str,
        "category_id": int,
        "split": str,
        **{n: float for n in BBOX_COLUMN_NAMES},
    }
    booleanized_columns: dict[str, set[str]] = {"images": set(), "annotations": set()}

    def __init__(
        self,
        images_root: Path | None = None,
        images: pd.DataFrame | None = None,
        annotations: pd.DataFrame | None = None,
        label_map: dict[int, str] | None = None,
        dataset_name: str | None = None,
    ):
        """Main Constructor

        Args:
            images_root: root path from where the ``relative_path`` values are relative
                to, in images
            images: DataFrame comprising image data. This dataframe should be referred
                to by annotations with the ``image_id`` column
            annotations: DataFrame comprising annotation data. Must have at least
                ``image_id`` column
            label_map: Mapping from ``category_id`` to ``category_str``, in the case the
                annotations have a ``category_id`` id. Useful for detections and
                classification
            dataset_name: Optional name for dataset. Will be used in function that need
                a name when the name cannot be easily deduced from images_root

        See Also:
            :meth:`from_template`

        Example:
            >>> Dataset()
            Dataset object containing 0 image and 0 object
            Name :
                None
            Images root :
                .
            Images :
            Empty DataFrame
            Columns: [width, height, relative_path, type]
            Index: []
            Annotations :
            Empty DataFrame
            Columns: [image_id, category_str, category_id, box_x_min, box_y_min, box_width, box_height]
            Index: []
            Label map :
            {}

            >>> images = pd.DataFrame(
            ...     data={
            ...         "width": [1920, 1280],
            ...         "height": [1080, 720],
            ...         "relative_path": [Path("0.jpg"), Path("1.jpg")],
            ...         "split": ["train", "valid"],
            ...     },
            ...     index=[0, 1],
            ... )
            >>> annotations = pd.DataFrame(
            ...     data={
            ...         "image_id": [0, 1],
            ...         "category_id": [1, 0],
            ...         "box_x_min": [10, 20],
            ...         "box_y_min": [30, 40],
            ...         "box_width": [100, 200],
            ...         "box_height": [200, 300],
            ...     },
            ...     index=[2, 3],
            ... )
            >>> label_map = {0: "this", 1: "that"}
            >>> Dataset(
            ...     images=images,
            ...     annotations=annotations,
            ...     label_map=label_map,
            ...     dataset_name="my_dataset",
            ... )
            Dataset object containing 2 images and 2 objects
            Name :
                my_dataset
            Images root :
                .
            Images :
                width  height relative_path  type  split
            id
            0    1920    1080         0.jpg  .jpg  train
            1    1280     720         1.jpg  .jpg  valid
            Annotations :
                image_id category_str  category_id  ... box_y_min  box_width  box_height
            id                                      ...
            2          0         that            1  ...      30.0      100.0       200.0
            3          1         this            0  ...      40.0      200.0       300.0
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {0: 'this', 1: 'that'}
        """
        if images_root is None:
            self.images_root = Path("")
        else:
            self.images_root = images_root
        if images is None:
            self.images = pd.DataFrame([], columns=list(self._image_required_columns))
        else:
            self.images = images
        self.init_images()

        # Note, although probably unnecessary, we do a full copy of annotation because
        # otherwise we get a warning from pandas. To be investigated, should potential
        # data fill the entire RAM one day
        if annotations is None:
            self.annotations = pd.DataFrame(
                [], columns=list(self._annotations_required_columns)
            )
        else:
            self.annotations = annotations.copy()

        if label_map is None:
            self.label_map = {}
        else:
            self.label_map = label_map
        self.booleanized_columns = {"images": set(), "annotations": set()}
        self.dataset_name = dataset_name
        self.init_annotations()

    def from_template(
        self,
        reset_booleanized: bool = False,
        **kwargs,
    ) -> Self:
        """Create a new Dataset object from an existing Dataset.

        Optionally, give new values for images_root, images, annotations or label map
        by providing supplementary kw arguments, which are to be fed to Dataset's
        ``__init__`` function.

        Note:
            - Although the Dataset object is a new one, dataframes are NOT cloned
            - booleanized columns are kept from other dataset to the new one.

        Args:
            reset_booleanized: If set to True, will reset booleanized columns for
                changed dataframes (and only for changed dataframes).
                Otherwise, the self.booleanized_columns dictionary of sets will only be
                updated so that columns that are not present anymore will be removed.
                Defaults to False
            **kwargs: keywords to overwrite other dataset's data with other values in
                the called constructor

        Returns:
            Resulting dataset, constructed from other dataset's data and optional
            additional data.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=0)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...  73.932999   71.552480   42.673983
            1          0          why           19  ...   4.567638  248.551257  122.602211
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

            >>> annotations = pd.DataFrame(
            ...     data={
            ...         "image_id": [0, 1],
            ...         "category_id": [12, 21],
            ...         "box_x_min": [10, 20],
            ...         "box_y_min": [30, 40],
            ...         "box_width": [100, 200],
            ...         "box_height": [200, 300],
            ...     },
            ...     index=[2, 3],
            ... )
            >>> Dataset.from_template(example, annotations=annotations)
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ... box_y_min  box_width  box_height
            id                                      ...
            2          0           12           12  ...      30.0      100.0       200.0
            3          1           21           21  ...      40.0      200.0       300.0
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {12: '12', 15: 'step', 19: 'why', 21: '21', 25: 'interview'}
        """
        booleanized_columns = deepcopy(self.booleanized_columns)
        if "images_root" not in kwargs:
            kwargs["images_root"] = self.images_root
        if "images" not in kwargs:
            kwargs["images"] = self.images
        elif reset_booleanized:
            booleanized_columns["images"] = set()
        if "annotations" not in kwargs:
            kwargs["annotations"] = self.annotations
        elif reset_booleanized:
            booleanized_columns["annotations"] = set()
        if "label_map" not in kwargs:
            kwargs["label_map"] = self.label_map
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = self.dataset_name

        DatasetSubclass = type(self)
        output_dataset = DatasetSubclass(**kwargs)

        updated_booleanized_columns = {"images": set(), "annotations": set()}
        for name, frame in zip(
            ["images", "annotations"],
            [output_dataset.images, output_dataset.annotations],
        ):
            for prefix in booleanized_columns[name]:
                try:
                    columns = get_bool_columns(frame, prefix)
                    if columns:
                        updated_booleanized_columns[name].add(prefix)
                except ValueError as e:
                    warn(
                        f"Prefix {prefix} will be ignored from generated dataset's"
                        f" {name} booleanized columns because of the following error:"
                        f" \n{e}",
                        RuntimeWarning,
                    )
        output_dataset.booleanized_columns = updated_booleanized_columns
        return output_dataset

    def rename(self, dataset_name: str) -> Self:
        """Simple function to change the name fo the dataset.

        The dataset name is used when printing it, showing it in jupyter or exporting
        it in other formats such as fiftyone.

        Equivalent to ``my_dataset.dataset_name = "new_name"``, but creates a new
        dataset instance (without copying the dataframes). It can be useful when using
        method chaining.

        Args:
            dataset_name: Name to give to the dataset.

        Returns:
            Renamed dataset

        Example:
            >>> Dataset().rename("my dataset")
            Dataset object containing 0 image and 0 object
            Name :
                my dataset
            Images root :
                .
            Images :
            Empty DataFrame
            Columns: [width, height, relative_path, type]
            Index: []
            Annotations :
            Empty DataFrame
            Columns: [image_id, category_str, category_id, box_x_min, box_y_min, box_width, box_height]
            Index: []
            Label map :
            {}

            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.loc[example.images["type"] == ".jpg"].rename("only_jpeg")
            Dataset object containing 1 image and 1 object
            Name :
                only_jpeg
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

        """
        return self.from_template(dataset_name=dataset_name)

    @property
    def loc(self) -> "DatasetImLocator[Self]":
        """Filter a dataset by indexing the images you want with their ids

        Similar to :attr:`pandas.DataFrame.loc` for images, but will create a new
        Dataset object and filter annotations accordingly.

        Note:
            You cannot set item with this method the same way you can in pandas

        Returns:
            Locator with a ``[]`` functionality relative to image id

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Image-based-sampling>`
            - :class:`.indexing.DatasetImLocator`
            - :attr:`iloc`
            - :meth:`filter_images`
            - :attr:`loc_annot`
            - :attr:`iloc_annot`
            - :meth:`filter_annotations`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.loc[example.images["type"] == ".jpg"]
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from .indexing import DatasetImLocator

        return DatasetImLocator(self, mode="loc")

    @property
    def iloc(self) -> "DatasetImLocator[Self]":
        """Filter a dataset by indexing the images you want with their row number.

        Similar to :attr:`pandas.DataFrame.iloc` for images, but will create a new
        Dataset object and filter annotations accordingly.

        Note:
            You cannot set item with this method the same way you can in pandas

        Returns:
            Locator with a ``[]`` functionality relative to image row

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Image-based-sampling>`
            - :class:`.indexing.DatasetImLocator`
            - :attr:`loc`
            - :meth:`filter_images`
            - :attr:`loc_annot`
            - :attr:`iloc_annot`
            - :meth:`filter_annotations`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.iloc[0]
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from .indexing import DatasetImLocator

        return DatasetImLocator(self, mode="iloc")

    @property
    def loc_annot(self) -> "DatasetAnnotLocator[Self]":
        """Filter a dataset by indexing the annotations you want with their id.

        Similar to :attr:`pandas.DataFrame.loc` for annotations, but will create a new
        Dataset object

        Note:
            - You cannot set item with this method the same way you can in pandas
            - Images emptied of annotation are NOT removed. If you want to remove
              emptied images, :meth:`.Dataset.filter_annotations` is better suited.

        Returns:
            Locator with a ``[]`` functionality relative to annotations id


        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Annotation-based-sampling>`
            - :class:`.indexing.DatasetAnnotLocator`
            - :attr:`loc`
            - :attr:`iloc`
            - :meth:`filter_images`
            - :attr:`iloc_annot`
            - :meth:`filter_annotations`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.loc_annot[example.annotations["box_height"] > 180]
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from .indexing import DatasetAnnotLocator

        return DatasetAnnotLocator(self, mode="loc")

    @property
    def iloc_annot(self) -> "DatasetAnnotLocator[Self]":
        """Filter a dataset by indexing the annotations you want with their row number.

        Similar to :attr:`pandas.DataFrame.iloc` for annotations, but will create a new
        Dataset object

        Note:
            - You cannot set item with this method the same way you can in pandas
            - Images emptied of annotation are NOT removed. If you want to remove
              emptied images, :meth:`.Dataset.filter_annotations` is better suited.

        Returns:
            Locator with a ``[]`` functionality relative to annotations row

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Annotation-based-sampling>`
            - :class:`.indexing.DatasetAnnotLocator`
            - :attr:`loc`
            - :attr:`iloc`
            - :meth:`filter_images`
            - :attr:`loc_annot`
            - :meth:`filter_annotations`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.iloc_annot[0]
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from .indexing import DatasetAnnotLocator

        return DatasetAnnotLocator(self, mode="iloc")

    def filter_images(self, index: Any, mode: Literal["loc", "iloc"] = "loc") -> Self:
        """Method equivalent of :attr:`.Dataset.loc` and :attr:`.Dataset.iloc`

        Args:
            index: Index object used in ``self.images.loc[]`` or ``self.images.iloc[]``
            mode: whether to be equivalent to :attr:`.Dataset.loc` or
                :attr:`.Dataset.iloc`. Defaults to "loc"

        Returns:
            Filtered dataset

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Image-based-sampling>`
            - :class:`.indexing.DatasetImLocator`
            - :attr:`loc`
            - :attr:`iloc`
            - :attr:`loc_annot`
            - :attr:`iloc_annot`
            - :meth:`filter_annotations`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.filter_images(example.images["type"] == ".jpg", mode="loc")
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.filter_images(0, mode="iloc")
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        if mode == "loc":
            return self.loc[index]
        else:
            return self.iloc[index]

    def filter_annotations(
        self,
        index: Any,
        mode: Literal["loc", "iloc"] = "loc",
        remove_emptied_images: bool = False,
    ) -> Self:
        """Method equivalent of :attr:`loc_annot` and
        :attr:`iloc_annot`, except you can choose to remove emptied images as
        well.

        Args:
            index: Index object used in ``self.annotations.loc[]`` or
                ``self.annotations.iloc[]``
            mode: whether to be equivalent to :meth:`.Dataset.loc_annot` or
                :attr:`.Dataset.iloc_annot`. Default to "loc"
            remove_emptied_images: if set to True, will remove images that were
                initially with annotations, but are now empty. In that case, it will
                keep the images that were already empty before calling this method.
                Default to False.

        Returns:
            Filtered dataset

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Annotation-based-sampling>`
            - :class:`.indexing.DatasetAnnotLocator`
            - :attr:`loc`
            - :attr:`iloc`
            - :meth:`filter_images`
            - :attr:`loc_annot`
            - :attr:`iloc_annot`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.filter_annotations(example.annotations["box_height"] > 180)
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.filter_annotations(0, mode="iloc")
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.filter_annotations(0, mode="iloc", remove_emptied_images=True)
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height   relative_path  type  split
            id
            1     131     840  air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from .indexing import DatasetAnnotLocator

        indexer = DatasetAnnotLocator(
            self, mode=mode, remove_emptied_images=remove_emptied_images
        )
        return indexer[index]

    def empty_annotations(self) -> Self:
        """Create a dataset object with an empty annotation dataframe, but with the same
        columns, and the same images dataframe.

        Useful when trying to construct a prediction dataset from another dataset

        Returns:
            New dataset instance with the same images as the original dataset, but an
            empty annotation dataframe

        See Also:
            - :meth:`filter_annotations`
            - :attr:`loc_annot`
            - :attr:`iloc_annot`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.empty_annotations()
            Dataset object containing 2 images and 0 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
            Empty DataFrame
            Columns: [image_id, category_str, category_id, split, box_x_min, box_y_min, box_width, box_height]
            Index: []
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        return self.iloc_annot[[]]

    def init_images(self):
        """Initialize images by checking required fields are present and converting
        fields to the right dtype. Also reorder columns so that required columns are
        first and tags last
        """
        from ..utils.dataframe_formatter import reorder_columns, set_dataframe_dtypes
        from ..utils.testing import assert_required_columns_present

        assert_required_columns_present(
            self.images, self._image_required_columns, df_name="images"
        )

        if self.images.index.has_duplicates:
            raise ValueError(
                "Dataset images ids are not exclusive, it will create ambiguity for"
                " annotation"
            )

        self.images.index.name = "id"
        self.images = self.images.assign(
            relative_path=self.images["relative_path"].apply(Path)  # pyright: ignore
        )

        if "type" not in self.images.columns:
            self.images["type"] = self.images["relative_path"].apply(lambda x: x.suffix)

        self.images = set_dataframe_dtypes(
            self.images,
            self._default_image_columns_with_types,
            nullable_types=["split"],
        )
        self.images = reorder_columns(
            self.images, list(self._default_image_columns_with_types.keys()), "."
        )

    def init_annotations(self):
        """Initialize annotations by adding info and checking index

        - add ``category_str`` column (for informative purpose only, label map prevails)
        - add ``split`` column (for informative purpose only, images split prevails)
        - reset index if it has duplicates.
        - apply the right dtypes
        - reorder the columns so that required columns are first and attributes last
        """
        from ..utils.dataframe_formatter import reorder_columns, set_dataframe_dtypes
        from ..utils.testing import assert_required_columns_present

        assert_required_columns_present(
            self.annotations,
            required_columns=self._annotations_required_columns,
            df_name="annotations",
        )

        valid_image_ids = self.annotations["image_id"].isin(self.images.index)
        if not valid_image_ids.all():
            wrong_ids = (
                self.annotations.loc[~valid_image_ids, "image_id"].unique().tolist()
            )
            raise ValueError(
                "The following image ids are not present in the dataset's images"
                f" dataframe: {', '.join(wrong_ids)}"
            )

        self.annotations.index.name = "id"
        all_cat_ids = set(self.annotations["category_id"].unique())
        if not all_cat_ids.issubset(self.label_map):
            missing_ids = all_cat_ids - self.label_map.keys()
            warn(
                "Incomplete Label map, setting following label of the following id to"
                f" their string equivalent : {missing_ids}",
                RuntimeWarning,
            )
            for i in missing_ids:
                self.label_map[i.item()] = str(i)

        self.annotations["category_str"] = (
            self.annotations["category_id"].astype(object).replace(self.label_map)
        )
        if "split" in self.images.columns:
            self.annotations["split"] = self.images.loc[
                self.annotations["image_id"], "split"
            ].values
        self.annotations = set_dataframe_dtypes(
            self.annotations,
            self._default_annotation_columns_with_types,
            nullable_types=["split"],
        )
        if self.annotations.index.has_duplicates:
            warn(
                "Dataset annotations have duplicates ids, resetting them ...",
                RuntimeWarning,
            )
            self.annotations.index = pd.RangeIndex(len(self.annotations), name="id")

        self.annotations = reorder_columns(
            self.annotations,
            list(self._default_annotation_columns_with_types.keys()),
            ".",
        )
        # TODO maybe have a few checks on annotations integrity ?
        # Bboxes size, position, etc

    def reset_images_root(self, new_path: Path | str) -> Self:
        """Replace the images_root with a new path. Relative path to images are updated
        accordingly so that ``new_path/new_relative_path`` still point to the right
        path.

        Args:
            new_path: New path to replace current images_root with

        Returns:
            New dataset object with updated images (relative_path column) and
            images_root.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1, generate_real_images=True)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                /tmp/care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.check()
            Checking Image and annotations Ids ...
            Checking Bounding boxes ..
            Checking label map ...
            Checking images are valid ...
            >>> example = example.reset_images_root("/tmp/")
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                /tmp
            Images :
                width  height                     relative_path  type  split
            id
            0     955     229  care/suggest/determine/story.jpg  .jpg  train
            1     131     840       care/suggest/air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.check()
            Checking Image and annotations Ids ...
            Checking Bounding boxes ..
            Checking label map ...
            Checking images are valid ...
        """
        new_path = Path(new_path)
        new_path_absolute = new_path.absolute()
        relative_path = Path(normpath(relpath(self.images_root, new_path)))
        new_image_paths = self.images["relative_path"].apply(
            lambda x: Path(  # pyright: ignore
                normpath(
                    relpath(
                        new_path_absolute / relative_path / x,
                        new_path_absolute,
                    )
                )
            )
        )
        new_images = self.images.assign(relative_path=new_image_paths)
        return self.from_template(
            images=new_images, images_root=new_path, reset_booleanized=False
        )

    def check(
        self,
        check_symlink: bool = False,
        allow_keypoints: bool = False,
        check_exhaustive: bool = False,
    ):
        """Make a full check of dataset, Ids, Bounding boxes, label maps and images

        See Also:
            :func:`.full_check_dataset_detection`

        Args:
            check_symlink: Whether the dataset should be using symlinks.
                Defaults to False.
            allow_keypoints: Whether a bounding box with a width and height of 0
                is acceptable and assumed to be a keypoint
            check_exhaustive: If set to True, will check that all images in the
                images_root folder are in the image dataframe, and that the dataset is
                indeed exhaustive
        """
        from lours.utils.testing import full_check_dataset_detection

        full_check_dataset_detection(
            self,
            check_symlink=check_symlink,
            allow_keypoints=allow_keypoints,
            check_exhaustive=check_exhaustive,
        )

    def remove_invalid_images(self, load_images: bool = True) -> Self:
        """Remove invalid images from dataset.

        See Also:
            - :func:`.get_invalid_images`
            - :meth:`.remove_invalid_annotations`

        Args:
            load_images: If set to True, will not only check that images are valid
                files, but also that image can be loaded (i.e. are not corrupted files)
                and that their sizes match the ones included in ``images``
                dataframe. Note that this makes the function significantly slower.
                Defaults to True.

        Returns:
            The same dataset, without the invalid images and their related annotations.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1, generate_real_images=True)
            >>> example.images.loc[0, "relative_path"] = Path("bad_path.jpg")
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                /tmp/care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229         bad_path.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.remove_invalid_images()
            Removed 1 image, with 1 annotation
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                /tmp/care/suggest
            Images :
                width  height   relative_path  type  split
            id
            1     131     840  air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
        """
        from lours.utils.testing import get_invalid_images

        invalid_images = get_invalid_images(
            self,
            check_symlink=False,
            load_images=load_images,
            raise_if_error=False,
        )

        n_invalid_images = len(invalid_images)
        n_invalid_annotations = self.loc[invalid_images.index].len_annot()
        print(
            f"Removed {n_invalid_images} image{'s' if n_invalid_images > 1 else ''},"
            " with"
            f" {n_invalid_annotations} annotation{'s' if n_invalid_annotations > 1 else ''}"
        )
        return self.loc[~self.images.index.isin(invalid_images.index)]

    def remove_invalid_annotations(
        self,
        allow_keypoints: bool = False,
        remove_related_images: bool = False,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Remove Invalid annotations from dataset.

        Optionally, remove images that have at least one invalid annotation, or remove
        images that have only invalid annotations

        See Also:
            - :func:`.get_malformed_bounding_boxes`
            - :meth:`.filter_annotations`
            - :meth:`.remove_invalid_images`

        Args:
            allow_keypoints: If set to True, will keep keypoints, i.e. bounding box
                with height and width of 0. Otherwise, will remove them.
                Defaults to False.
            remove_related_images: If set to True, will remove any image that has an
                invalid annotation. Defaults to False.
            remove_emptied_images: If set to True, will remove images that are empty
                after removing the invalid annotations. In other word, remove images
                where all annotations are invalid. Note that already empty images
                are not removed. Defaults to False.

        Returns:
            The same dataset, without the invalid annotations and optionally without
            their related and/or emptied images.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 4, seed=1)
            >>> example.annotations.loc[0, "box_width"] = -1
            >>> example
            Dataset object containing 2 images and 4 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg   eval
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1     marriage           15  ...  276.974642   -1.000000  353.331683
            1          0       listen           14  ...   64.213606  358.653949  116.336568
            2          0        reach           22  ...   69.431616  525.305264   41.677117
            3          1       listen           14  ...  380.938227   36.133726  442.881021
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.remove_invalid_annotations()
            Removed 1 annotation, in 1 image
            Dataset object containing 2 images and 3 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg   eval
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            1          0       listen           14  ...   64.213606  358.653949  116.336568
            2          0        reach           22  ...   69.431616  525.305264   41.677117
            3          1       listen           14  ...  380.938227   36.133726  442.881021
            <BLANKLINE>
            [3 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.remove_invalid_annotations(remove_related_images=True)
            Removed 1 image with invalid annotations
            Dataset object containing 1 image and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type split
            id
            0     955     229  determine/story.jpg  .jpg  eval
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            1          0       listen           14  ...  64.213606  358.653949  116.336568
            2          0        reach           22  ...  69.431616  525.305264   41.677117
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 4, seed=1)
            >>> example.annotations.loc[[0, 3], "box_width"] = -1
            >>> example
            Dataset object containing 2 images and 4 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg   eval
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1     marriage           15  ...  276.974642   -1.000000  353.331683
            1          0       listen           14  ...   64.213606  358.653949  116.336568
            2          0        reach           22  ...   69.431616  525.305264   41.677117
            3          1       listen           14  ...  380.938227   -1.000000  442.881021
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.remove_invalid_annotations(remove_emptied_images=True)
            Removed 2 annotations, in 1 image
            Dataset object containing 1 image and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type split
            id
            0     955     229  determine/story.jpg  .jpg  eval
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            1          0       listen           14  ...  64.213606  358.653949  116.336568
            2          0        reach           22  ...  69.431616  525.305264   41.677117
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

        """
        from lours.utils.testing import get_malformed_bounding_boxes

        invalid_annots = get_malformed_bounding_boxes(
            self, allow_keypoints=allow_keypoints, raise_if_error=False
        )
        invalid_images = self.annotations.loc[invalid_annots.index, "image_id"].unique()

        if remove_related_images:
            n_images = len(invalid_images)
            print(
                f"Removed {n_images} image{'s' if n_images > 1 else ''} with invalid"
                " annotations"
            )
            return self.loc[~self.images.index.isin(invalid_images)]
        else:
            n_annots = len(invalid_annots)
            n_images = len(invalid_images)
            print(
                f"Removed {n_annots} annotation{'s' if n_annots > 1 else ''}, in"
                f" {n_images} image{'s' if n_images > 1 else ''}"
            )
            return self.filter_annotations(
                ~self.annotations.index.isin(invalid_annots.index),
                remove_emptied_images=remove_emptied_images,
            )

    def get_one_frame(self, n: int) -> tuple[pd.Series, pd.DataFrame]:
        """Sample a single image from the dataset. Image data is returned as a pandas
        Series, and corresponding annotations is returned as a DataFrame.

        This equivalent to ``dataset.iloc[n]`` except the returned object is the bare
        image info and annotation dataframe. This can be useful when using lours as e.g.
        a pytorch dataset.

        Note:
            The id of the image is the name of the image Series

        Args:
            n: row number of wanted image. Note that this does NOT use the index of
                self.images.

        Returns:
            tuple containing image data as Series and annotations as a (possibly empty)
            DataFrame.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Iterating-through-the-dataset>`
            - :meth:`iter_images`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> frame, annotations = example.get_one_frame(0)
            >>> frame
            width                            955
            height                           229
            relative_path    determine/story.jpg
            type                            .jpg
            split                          train
            Name: 0, dtype: object
            >>> annotations
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]

        """
        image_data = self.images.iloc[n]
        annotations = self.annotations[
            self.annotations["image_id"] == self.images.index[n]
        ]
        return image_data, annotations

    def iter_images(self) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
        """Iterate through images, by yielding

        Yields:
            tuple containing:
             - image Series with image data, and named as the image id
             - annotations DataFrame

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Iterating-through-the-dataset>`
            - :meth:`get_one_frame`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> for i, (frame, annot) in enumerate(example.iter_images()):
            ...     print(f"Frame {i}")
            ...     print(frame)
            ...     print(annot)
            ...
            Frame 0
            width                            955
            height                           229
            relative_path    determine/story.jpg
            type                            .jpg
            split                          train
            Name: 0, dtype: object
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Frame 1
            width                       131
            height                      840
            relative_path    air/method.bmp
            type                       .bmp
            split                     train
            Name: 1, dtype: object
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
        """
        for i in range(len(self)):
            yield self.get_one_frame(i)

    def get_image_attributes(self) -> list[str]:
        """Get the name of columns related to image attributes. In other words, get
        columns that are NOT the default ones.

        The actual attribute values can then be
        ``self.images[self.get_image_attributes()]``

        Returns:
            list of column names in ``self.images`` that represent tags

        See Also:
            :meth:`get_annotations_attributes`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example.images["something"] = True
            >>> example.images["else"] = 10
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split  something  else
            id
            0     955     229  determine/story.jpg  .jpg  train       True    10
            1     131     840       air/method.bmp  .bmp  train       True    10
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.get_image_attributes()
            ['something', 'else']
        """
        return [
            str(c)
            for c in self.images.columns
            if c not in self._default_image_columns_with_types.keys()
        ]

    def get_annotations_attributes(self) -> list[str]:
        """Get the name of columns related to annotations attributes. In other words,
        get columns that are NOT the default ones.

        the actual attribute values can then be
        ``self.annotations[self.get_annotations_attributes()]``

        Returns:
            list of column names in ``self.annotations`` that represent attributes

        See Also:
            :meth:`get_image_attributes`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example.annotations["else"] = 10
            >>> example.annotations["something"] = True
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...  box_height  else  something
            id                                      ...
            0          1       listen           14  ...  184.684056    10       True
            1          0        reach           22  ...  174.239136    10       True
            <BLANKLINE>
            [2 rows x 10 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.get_annotations_attributes()
            ['else', 'something']
        """
        return [
            str(c)
            for c in self.annotations.columns
            if c not in self._default_annotation_columns_with_types.keys()
        ]

    def __getitem__(self, args: Any) -> Self:
        """``__getitem__`` implementation for the Dataset object. The iteration is made
        image wise. Constructs a sub dataset so that the image index of that new
        dataset is the result of ``self.images[args]``. ``args`` could be anything like
        slices, ellipsis and so on.

        Note:
            This is equivalent to calling the dataset indexer :attr:`.Dataset.iloc`

        Args:
            args: usual parameters for indexing or slicing a numpy array or a pandas
                array with ``iloc``. This will be used to index image indices for the
                returned dataset object.

        Returns:
            Sub-dataset including image data indices and corresponding annotations
        """
        return self.iloc[args]

    def __len__(self) -> int:
        """Return number of images in dataset.
        to get number of annotations,
        use the method :meth:`.Dataset.len_annot`

        Returns:
            Length of ``self.images`` dataframe
        """
        return len(self.images)

    def len_annot(self) -> int:
        """Return number of annotations in total

        Returns:
            Length of ``self.annotations`` dataframe
        """
        return len(self.annotations)

    def __bool__(self) -> bool:
        return len(self.images) > 0

    def _description(self) -> str:
        images_word = "images" if len(self) > 1 else "image"
        annotations_word = "objects" if self.len_annot() > 1 else "object"
        return (
            f"Dataset object containing {len(self):,} {images_word} "
            f"and {self.len_annot():,} {annotations_word}\n"
            f"Name :\n\t{self.dataset_name}\n"
            f"Images root :\n\t{self.images_root}"
        )

    def __repr__(self) -> str:
        return (
            f"{self._description()}\n"
            f"Images :\n{self.images}\n"
            f"Annotations :\n{self.annotations}\n"
            f"Label map :\n{pformat(self.label_map)}"
        )

    def _ipython_display_(self):
        """Function to display the Dataset as an HTML widget when using notebooks"""
        import ipywidgets as widgets
        from ipykernel.zmqshell import ZMQInteractiveShell
        from IPython.core.getipython import get_ipython
        from IPython.display import display

        from ..utils.notebook_utils import display_booleanized_dataframe

        is_notebook = isinstance(get_ipython(), ZMQInteractiveShell)
        if not is_notebook:
            print(self)
            return

        tab = widgets.Tab()

        descr_str = (
            "<p><span style='white-space: pre-wrap; font-weight:"
            f" bold'>{self._description()}</span></p>"
        )

        title = widgets.HTML(descr_str)

        label_map_df = (
            pd.Series(self.label_map, name="category string").to_frame().sort_index()
        )
        label_map_df.index.name = "category_id"

        # create output widgets
        widget_images = widgets.Output()
        widget_annotations = widgets.Output()
        widget_label_map = widgets.Output()

        # render in output widgets
        with widget_images:
            display_booleanized_dataframe(
                self.images, self.booleanized_columns["images"]
            )
        with widget_annotations:
            display_booleanized_dataframe(
                self.annotations, self.booleanized_columns["annotations"]
            )
        with widget_label_map:
            display(label_map_df)

        tab.children = [widget_images, widget_annotations, widget_label_map]
        tab.titles = ["Images", "Annotations", "Label Map"]

        display(widgets.VBox([title, tab]))

    def get_split(self, split: str | None) -> Self:
        """Get a particular split from the dataset

        Args:
            split: Split name, usually "train", "val", "eval". If set to None, will
                retrieve all image with a null split value (None, pd.NA or np.nan)

        Returns:
            filtered dataset, with only samples within the wanted split

        See Also:
            :meth:`iter_splits`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(
            ...     2,
            ...     2,
            ...     split_names=["train", "eval"],
            ...     split_shares=[0.5, 0.5],
            ...     seed=14,
            ... )
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                present_wait_even
            Images root :
                blood/reflect
            Images :
                width  height      relative_path   type  split
            id
            0     424     732  listen/reason.bmp   .bmp  train
            1     179     413    return/man.jpeg  .jpeg   eval
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1         film           24  ...  192.940695    2.862400   74.219110
            1          0   especially            4  ...  419.039943  276.766197  119.753886
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {4: 'especially', 19: 'similar', 24: 'film'}
            >>> example.get_split("eval")
            Dataset object containing 1 image and 1 object
            Name :
                present_wait_even
            Images root :
                blood/reflect
            Images :
                width  height    relative_path   type split
            id
            1     179     413  return/man.jpeg  .jpeg  eval
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1         film           24  ...  192.940695     2.8624    74.21911
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {4: 'especially', 19: 'similar', 24: 'film'}
        """
        if "split" not in self.images.columns:
            warn("Dataset has no split value", RuntimeWarning)
            return self.loc[[]]
        if split is not None:
            split_image_ids = self.images["split"] == split
        else:
            split_image_ids = self.images["split"].isnull()
        return self.loc[split_image_ids]

    def iter_splits(self) -> Iterator[tuple[str | None, Self]]:
        """Iterate though split values of the dataset, by yielding for each split
        the split name and the corresponding sub-dataset.

        If no split is available, the split value is assumed to be ``None`` for the
        whole dataset.

        Yields:
            tuple containing:
             - the name of the split
             - the corresponding subset of the original dataset

        See Also:
            :meth`get_split`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=2)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height      relative_path   type  split
            id
            0     368     832  police/enter.jpeg  .jpeg  train
            1     472     506    also/policy.gif   .gif  train
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          0         table            7  ...  228.774514  137.766169  131.174304
            1          0  relationship            3  ...  546.984268   34.928954    9.871084
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
            >>> for split_name, split in example.iter_splits():
            ...     print(f"Split: {split_name}")
            ...     print(split)
            ...
            Split: train
            Dataset object containing 2 images and 2 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height      relative_path   type  split
            id
            0     368     832  police/enter.jpeg  .jpeg  train
            1     472     506    also/policy.gif   .gif  train
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          0         table            7  ...  228.774514  137.766169  131.174304
            1          0  relationship            3  ...  546.984268   34.928954    9.871084
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
        """
        if "split" not in self.images.columns:
            yield None, self
        for split_value in self.images["split"].unique():
            yield split_value, self.get_split(split_value)

    def reset_index(
        self,
        start_image_id: int = 0,
        start_annotations_id: int = 0,
        sort_images_by: None | str | Sequence[str] = "relative_path",
        sort_annotations_by: None | str | Sequence[str] = (
            "image_id",
            "category_id",
            *BBOX_COLUMN_NAMES,
        ),
    ) -> Self:
        """Reset index of ``self.images`` dataframe, and reset index of self.annotations
        However, keep the 'image_id' column in ``self.annotations`` pointing to the
        right rows in the ``self.images`` dataframe.

        Note:
            Both images and annotations dataframes will be reorder according to specific
            columns. You can change them with the ``sort_images_by`` and
            ``sort_annotations_by`` parameters, but the default behaviour is:

            - images dataframe will be reordered according to ``relative_path``
            - annotations dataframe will be reordered according to ``image_id``,
              ``category_id`` and the bounding box coordinates, i.e. ``box_x_min``,
              ``box_y_min``, ``box_width`` and ``box_height``

        Args:
            start_image_id: Number at which the image index starts. This is used to
                construct two datasets without overlapping ids.
            start_annotations_id: Similar to start_image_id, number at which the
                annotations index starts.
            sort_images_by: columns to sort the images dataframe by. It is advised to
                chose a collection of columns that makes the sorting unique. If set to
                None or an empty sequence, will no sort the images dataframe before
                applying a range index to it. Defaults to ``relative_path``
            sort_annotations_by: columns to sort the annotations dataframe by. It is
                advised to chose a collection of columns that makes the sorting unique.
                If set to None or an empty sequence, will not sort the annotations
                dataframe before applying a range index to it. Defaults to
                ``("image_id", "category_id", "box_x_min",
                "box_y_min", "box_width", box_height")``.

        Returns:
            Dataset with ``self.images`` and ``self.annotations`` with updated indexes

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Resetting-index>`
            - :meth:`reset_index_from_mapping`
            - :meth:`match_index`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(10, 10, seed=2)
            >>> example.iloc[1::2]
            Dataset object containing 5 images and 4 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height          relative_path   type  split
            id
            1     472     892        also/policy.gif   .gif  train
            3     506     602      increase/pull.jpg   .jpg    val
            5     401     281         would/off.jpeg  .jpeg  train
            7     831     375        ahead/truth.bmp   .bmp  train
            9     993     334  husband/whatever.jpeg  .jpeg   eval
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          1  relationship            3  ...  606.391824   29.194750  194.387036
            1          7  relationship            3  ...  313.193702  230.609055    5.269920
            5          9        simply           25  ...  198.210135  474.192703   57.594892
            9          9         table            7  ...   60.522880  425.022919  144.458578
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
            >>> example.iloc[1::2].reset_index(10, 5)
            Dataset object containing 5 images and 4 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height          relative_path   type  split
            id
            10    831     375        ahead/truth.bmp   .bmp  train
            11    472     892        also/policy.gif   .gif  train
            12    993     334  husband/whatever.jpeg  .jpeg   eval
            13    506     602      increase/pull.jpg   .jpg    val
            14    401     281         would/off.jpeg  .jpeg  train
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            5         10  relationship            3  ...  313.193702  230.609055    5.269920
            6         11  relationship            3  ...  606.391824   29.194750  194.387036
            7         12         table            7  ...   60.522880  425.022919  144.458578
            8         12        simply           25  ...  198.210135  474.192703   57.594892
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
        """
        if sort_images_by is None:
            sort_images_by = []
        elif isinstance(sort_images_by, str):
            sort_images_by = [sort_images_by]
        else:
            sort_images_by = [*sort_images_by]
        if sort_annotations_by is None:
            sort_annotations_by = []
        elif isinstance(sort_annotations_by, str):
            sort_annotations_by = [sort_annotations_by]
        else:
            sort_annotations_by = [*sort_annotations_by]
        if len(sort_images_by) > 0:
            new_images = self.images.sort_values(sort_images_by)
        else:
            new_images = self.images

        new_images = new_images.assign(
            new_id=np.arange(start_image_id, start_image_id + len(self))
        )

        new_annotations = self.annotations.assign(
            image_id=new_images.loc[self.annotations["image_id"], "new_id"].to_numpy()
        )

        if len(sort_annotations_by) > 0:
            new_annotations = new_annotations.sort_values(
                sort_annotations_by
            ).reset_index(drop=True)
        new_annotations.index.name = "id"
        new_annotations.index += start_annotations_id
        new_images = new_images.set_index("new_id")
        new_images.index.name = "id"

        return self.from_template(images=new_images, annotations=new_annotations)

    def reset_index_from_mapping(
        self,
        images_index_map: dict[int, int] | pd.DataFrame | pd.Series | None = None,
        annotations_index_map: dict[int, int] | pd.DataFrame | pd.Series | None = None,
        remove_unmapped: bool = False,
    ) -> Self:
        """Reset index of images and annotations dataframe with index maps
        (index -> new_index) where the value is new index to apply.

        The mapping can be either a dictionary, a pandas Series or a DataFrame with
        only one column. If the dataframe has more than 1 column, this function will
        raise an error

        Args:
            images_index_map: Mapping from original image index to new image index.
                If it is a DataFrame, it must have only one column.
                If set to None, will apply the identity mapping. Defaults to None.
            annotations_index_map: Mapping. Same as ``images_index_map``, but this
                mapping applies for annotations. If set to None, will apply the identity
                mapping. Default to None.
            remove_unmapped: If set to True, will remove the entries in the original
                dataframes which index is not present in the given mappings. Otherwise,
                will apply a default mapping so that it is bijective. A range index
                starting at the highest mapped index+1 will be applied to the missing
                values in the mapping index. Defaults to False.

        Returns:
            Dataset: new dataset instance with images and annotations dataframes which
            index have been remapped. The annotations will be filtered out according to
            removed images, and its "image_id" column will be modified to match the new
            image index.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Reindex-with-mapping>`
            - :meth:`reset_index`
            - :meth:`match_index`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(3, 3, seed=2)
            >>> example
            Dataset object containing 3 images and 3 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height            relative_path   type  split
            id
            0     368     506        police/enter.jpeg  .jpeg  train
            1     472     182          also/policy.gif   .gif  train
            2     832     401  cold/responsibility.png   .png    val
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          1  relationship            3  ...   27.311332   69.768824   97.006466
            1          2        simply           25  ...  157.041558   20.174848   16.443389
            2          2  relationship            3  ...   75.088280  337.101681  193.299936
            <BLANKLINE>
            [3 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}

            Note that unmapped index gets remapped to a range index starting after the
            highest value of mapped index, hence the annotation id "2" that gets mapped
            to "3" even if index "1" was available.

            >>> example.reset_index_from_mapping(
            ...     images_index_map={0: 1, 2: 0}, annotations_index_map={1: 2, 2: 0}
            ... )
            Dataset object containing 3 images and 3 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height            relative_path   type  split
            id
            1     368     506        police/enter.jpeg  .jpeg  train
            0     832     401  cold/responsibility.png   .png    val
            2     472     182          also/policy.gif   .gif  train
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            2          0        simply           25  ...  157.041558   20.174848   16.443389
            0          0  relationship            3  ...   75.088280  337.101681  193.299936
            3          2  relationship            3  ...   27.311332   69.768824   97.006466
            <BLANKLINE>
            [3 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}

            >>> example.reset_index_from_mapping(
            ...     images_index_map={0: 1, 2: 0},
            ...     annotations_index_map={1: 2, 2: 0},
            ...     remove_unmapped=True,
            ... )
            Dataset object containing 2 images and 2 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height            relative_path   type  split
            id
            1     368     506        police/enter.jpeg  .jpeg  train
            0     832     401  cold/responsibility.png   .png    val
            Annotations :
                image_id  category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            2          0        simply           25  ...  157.041558   20.174848   16.443389
            0          0  relationship            3  ...   75.088280  337.101681  193.299936
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
        """

        def convert_mapping_to_series(
            input_mapping: dict[int, int] | pd.DataFrame | pd.Series | None,
            mapping_name: str,
        ) -> pd.Series | None:
            if isinstance(input_mapping, pd.DataFrame):
                if len(input_mapping.columns) > 1:
                    raise ValueError(
                        "Index mapping can only be a Series or a DataFrame with 1"
                        f" column. The mapping {mapping_name} got"
                        f" {len(input_mapping.columns)} columns instead"
                    )
                return input_mapping.iloc[:, 0]
            if isinstance(input_mapping, dict):
                input_mapping = pd.Series(input_mapping)
            return input_mapping

        images_index_map = convert_mapping_to_series(
            images_index_map, "images_index_map"
        )
        annotations_index_map = convert_mapping_to_series(
            annotations_index_map, "annotations_index_map"
        )

        def reindex_dataframe(
            input_df: pd.DataFrame,
            index_mapping: pd.Series | None,
            remove_unmapped: bool,
        ) -> tuple[pd.DataFrame, pd.Series | None]:
            if index_mapping is None:
                return input_df, None
            mapped_index = input_df.index.intersection(index_mapping.index)
            index_mapping = index_mapping.loc[mapped_index]
            if len(mapped_index) != len(input_df) and not remove_unmapped:
                unmapped = input_df.index.difference(
                    mapped_index,
                    sort=False,
                )
                residual_mapping = pd.Series(
                    np.arange(len(unmapped)) + index_mapping.max() + 1,
                    index=unmapped,
                )
                index_mapping = pd.concat([index_mapping, residual_mapping])
            return (
                input_df.loc[index_mapping.index].set_index(index_mapping),
                index_mapping,
            )

        new_annotations, _ = reindex_dataframe(
            self.annotations, annotations_index_map, remove_unmapped
        )
        new_images, images_index_map = reindex_dataframe(
            self.images, images_index_map, remove_unmapped
        )
        if images_index_map is not None:
            if remove_unmapped and len(images_index_map) < len(self):
                new_annotations = new_annotations[
                    new_annotations["image_id"].isin(images_index_map.index)
                ]
            new_annotations = new_annotations.assign(
                image_id=images_index_map.loc[new_annotations["image_id"]].values
            )
        return self.from_template(images=new_images, annotations=new_annotations)

    def match_index(
        self,
        other_images: "pd.DataFrame | Dataset",
        on: str = "relative_path",
        remove_unmatched: bool = False,
    ) -> Self:
        """Reindex a dataset from another images DataFrame.

        The given ``on`` column is used to retrieve the index values from the reference
        images dataframe.

        Note:
            If index of rows which value in ``on`` column does not match any row in
            ``other_images``, DataFrame's index will be reset to a range index without
            sorting it.

        Args:
            other_images: images DataFrame taken from another dataset. Must have the
                column specified in ``on``
            on: name of the column to use to retrieve indexes. Must be present in both
                columns of  ``self.images`` and ``other_images``.
                Defaults to "relative_path".
            remove_unmatched: if set to True, will remove images from dataset that don't
                match any row in the ``other_images`` dataframe. The corresponding
                annotations will also be removed.

        Returns:
            Dataset with updated image indexes, along with values in ``image_id`` column
            of annotations.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Reindex-images-index-from-other-dataframe>`
            - :meth:`reset_index`
            - :meth:`reset_index_from_mapping`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(5, 5, seed=2)
            >>> example
            Dataset object containing 5 images and 5 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height            relative_path   type  split
            id
            0     368     401        police/enter.jpeg  .jpeg  train
            1     472     640          also/policy.gif   .gif    val
            2     832     831  cold/responsibility.png   .png  train
            3     506     755        increase/pull.jpg   .jpg  train
            4     182     993            Mr/trade.tiff  .tiff  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0       simply           25  ...  273.908994  168.756932    4.288302
            1          4        table            7  ...  106.456857   19.340529  282.426602
            2          0       simply           25  ...   41.921967   38.506811   33.166314
            3          2        table            7  ...  167.785089  242.139038  119.708224
            4          1       simply           25  ...  327.082223  234.360304  238.965568
            <BLANKLINE>
            [5 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}
            >>> images_modified = example.images.iloc[::2].reset_index(drop=True)
            >>> images_modified
               width  height            relative_path   type  split
            0    368     401        police/enter.jpeg  .jpeg  train
            1    832     831  cold/responsibility.png   .png  train
            2    182     993            Mr/trade.tiff  .tiff  train
            >>> example.match_index(images_modified)
            Dataset object containing 5 images and 5 objects
            Name :
                argue_be_structure
            Images root :
                what/way
            Images :
                width  height            relative_path   type  split
            id
            0     368     401        police/enter.jpeg  .jpeg  train
            1     832     831  cold/responsibility.png   .png  train
            2     182     993            Mr/trade.tiff  .tiff  train
            3     472     640          also/policy.gif   .gif    val
            4     506     755        increase/pull.jpg   .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0       simply           25  ...  273.908994  168.756932    4.288302
            1          2        table            7  ...  106.456857   19.340529  282.426602
            2          0       simply           25  ...   41.921967   38.506811   33.166314
            3          1        table            7  ...  167.785089  242.139038  119.708224
            4          3       simply           25  ...  327.082223  234.360304  238.965568
            <BLANKLINE>
            [5 rows x 8 columns]
            Label map :
            {3: 'relationship', 7: 'table', 25: 'simply'}

        """
        if isinstance(other_images, Dataset):
            other_images = other_images.images
        other_on_index = pd.Index(other_images[on])
        self_on_index = pd.Index(self.images[on])

        if other_on_index.has_duplicates:
            raise ValueError(
                f"The column {on} of the input image dataframe has duplicate values"
            )
        if self_on_index.has_duplicates:
            raise ValueError(
                f"The column {on} of the dataset's image dataframe has duplicate values"
            )

        # Construct 2 Series indexed by the "on" anchor
        # So that we can align them and be able to telle which index in self images
        # corresponds to which index in the other images
        other_images_index_values = pd.Series(
            other_images.index, index=other_on_index, name="other_id"
        )
        index_values_to_match = pd.Series(
            self.images.index, index=self_on_index, name="self_id"
        )

        # Concatenante the series with the "inner" join to remove index values without
        # correspondence. By setting the index and selecting the column, we now have
        # a Series that models the original id -> new id mapping
        matched_ids_map = pd.concat(
            [other_images_index_values, index_values_to_match], join="inner", axis=1
        ).set_index("self_id")["other_id"]

        return self.reset_index_from_mapping(
            images_index_map=matched_ids_map, remove_unmapped=remove_unmatched
        )

    def merge(
        self,
        other: "Dataset",
        allow_overlapping_image_ids: bool = True,
        realign_label_map: bool = False,
        ignore_index: bool = False,
        mark_origin: bool = False,
        overwrite_origin: bool = False,
    ) -> "Dataset":
        """Merge two datasets and return a unique dataset object containing
        Samples from both. Result's images_root will be the common path of both
        datasets, and the image relative paths will be updated accordingly.
        Result's label map will be the superset of both label map,
        provided one is included in the other.

        Notes:
            - This function is also usable with the `+` operator
            - If possible, booleanized columns for images and annotations will be
              broadcast together.
              See :func:`lours.utils.column_booleanizer.broadcast_booleanization`
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
            other: Other dataset to merge with. This dataset must be
                compatible with the first one, i.e. one label map is included with the
                other, and image and annotation ids are mutually exclusives between
                datasets (unless `ignore_index` is False)
            allow_overlapping_image_ids: if set to True, will try to join images
                dataframes with overlapping ids. The whole rows (i.e. with values from
                columns present in both dataframes) must match, as well as
                the images_root. In that case, annotations with this image_id
                (from self or other) will be assumed to come from the same image.
                Defaults to True
            realign_label_map: If set to True, will try to remap classes of other
                dataset to match this dataset's label map, to avoid a potential error
                due to incompatible label maps.
            ignore_index: if set to True, will ignore overlapping ids
                for images and annotations and reset them. Will update the ``image_id``
                column in the annotations accordingly. Note that this option makes the
                former option useless. Defaults to False.
            mark_origin: If set to True, and if both datasets have a different name,
                will add two columns "origin" and "origin_id" for images and
                annotations dataframes, indicating respectively the name of the origin
                dataset, and its id in the original dataset. Defaults to True.
            overwrite_origin: If set to True, will overwrite already existing columns in
                input datasets dataframes. Otherwise, will only mark origin if it's not
                present. Defaults to False.

        Raises:
            ValueError: Error if the two datasets are incompatible (see above)

        Returns:
            Merged dataset.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Dataset-merge>`
            - :func:`merge.merge_datasets`
            - :meth:`Dataset.__add__`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example1 = dummy_dataset(2, 2, seed=0)
            >>> example2 = dummy_dataset(2, 2, seed=1)
            >>> example1
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...  73.932999   71.552480   42.673983
            1          0          why           19  ...   4.567638  248.551257  122.602211
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example2
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}

            Notice how the two label maps have overlapping index (the id 15)

            >>> example1 + example2
            Using the following class remapping dictionary :
            {14: 14, 15: 16, 22: 22}
            Dataset object containing 4 images and 4 objects
            Name :
                inside_else_memory+shake_effort_many
            Images root :
                .
            Images :
                width  height                     relative_path   type  split
            id
            0     342     136         such/serious/help/me.jpeg  .jpeg  train
            1     377     167    such/serious/whatever/wait.png   .png  train
            2     131     840       care/suggest/air/method.bmp   .bmp  train
            3     955     229  care/suggest/determine/story.jpg   .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...   73.932999   71.552480   42.673983
            1          0          why           19  ...    4.567638  248.551257  122.602211
            2          2       listen           14  ...  276.974642    9.718823  184.684056
            3          3        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {14: 'listen',
             15: 'step',
             16: 'marriage',
             19: 'why',
             22: 'reach',
             25: 'interview'}

            >>> example1.merge(example2, realign_label_map=False)
            Traceback (most recent call last):
                ...
            lours.utils.label_map_merger.IncompatibleLabelMapsError: Label maps are incompatible

            >>> example1.merge(
            ...     example2, realign_label_map=True, allow_overlapping_image_ids=False
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Overlapping image ids not permitted. Consider using the allow_overlapping_image_ids or ignore_index options

            This will raise an error because overlapping image ids is possible only if
            the rows are compatible : fields that are present in both rows have the
            same value

            >>> example1.merge(
            ...     example2, realign_label_map=True, allow_overlapping_image_ids=True
            ... )
            Traceback (most recent call last):
                ...
            AssertionError: sub-Dataframes constructed from ids and columns in both DataFrames are not equal.

            The only way to merge these datasets is to remap the label map and then
            reset the indexes with the option ``ignore_index`` set to ``True``, similar
            to :func:`pandas.concat`.

            >>> example1.merge(
            ...     example2.remap_classes({15: 1}, remove_not_mapped=False),
            ...     ignore_index=True,
            ... )
            Dataset object containing 4 images and 4 objects
            Name :
                inside_else_memory+shake_effort_many
            Images root :
                .
            Images :
                width  height                     relative_path   type  split
            id
            0     342     136         such/serious/help/me.jpeg  .jpeg  train
            1     377     167    such/serious/whatever/wait.png   .png  train
            2     131     840       care/suggest/air/method.bmp   .bmp  train
            3     955     229  care/suggest/determine/story.jpg   .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...   73.932999   71.552480   42.673983
            1          0          why           19  ...    4.567638  248.551257  122.602211
            2          2       listen           14  ...  276.974642    9.718823  184.684056
            3          3        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {1: 'marriage',
            14: 'listen',
            15: 'step',
            19: 'why',
            22: 'reach',
            25: 'interview'}

            Let's construct two datasets sharing image info and label maps

            >>> example = dummy_dataset(5, 5, seed=0)
            >>> example1 = example.iloc_annot[::2].iloc[1:]
            >>> example2 = example.iloc_annot[1::2].iloc[:-1]

            >>> example1
            Dataset object containing 4 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height           relative_path   type  split
            id
            1     377     831       whatever/wait.png   .png  train
            2     136     684        chair/mother.gif   .gif  train
            3     167     921  someone/challenge.jpeg  .jpeg  train
            4     114     553  successful/present.bmp   .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          3          why           19  ...  498.685784  31.192237  404.663563
            2          3    interview           25  ...  389.294931  19.083146  209.778063
            4          2         step           15  ...   85.009761  18.228218  181.012493
            <BLANKLINE>
            [3 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example2
            Dataset object containing 4 images and 1 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height           relative_path   type  split
            id
            0     342     257            help/me.jpeg  .jpeg  train
            1     377     831       whatever/wait.png   .png  train
            2     136     684        chair/mother.gif   .gif  train
            3     167     921  someone/challenge.jpeg  .jpeg  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min  box_width  box_height
            id                                      ...
            3          3         step           15  ...  26.082417  34.739663  607.977022
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example1.merge(example2)
            Dataset object containing 5 images and 4 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height           relative_path   type  split
            id
            1     377     831       whatever/wait.png   .png  train
            2     136     684        chair/mother.gif   .gif  train
            3     167     921  someone/challenge.jpeg  .jpeg  train
            4     114     553  successful/present.bmp   .bmp  train
            0     342     257            help/me.jpeg  .jpeg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          3          why           19  ...  498.685784  31.192237  404.663563
            2          3    interview           25  ...  389.294931  19.083146  209.778063
            4          2         step           15  ...   85.009761  18.228218  181.012493
            3          3         step           15  ...   26.082417  34.739663  607.977022
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

            See that if we use the ``ignore_index`` option, the images are duplicated
            because it is assumed the two images dataframes don't have any overlap.

            >>> example1.merge(example2, ignore_index=True)
            Dataset object containing 8 images and 4 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height           relative_path   type  split
            id
            0     136     684        chair/mother.gif   .gif  train
            1     167     921  someone/challenge.jpeg  .jpeg  train
            2     114     553  successful/present.bmp   .bmp  train
            3     377     831       whatever/wait.png   .png  train
            4     136     684        chair/mother.gif   .gif  train
            5     342     257            help/me.jpeg  .jpeg  train
            6     167     921  someone/challenge.jpeg  .jpeg  train
            7     377     831       whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          0         step           15  ...   85.009761  18.228218  181.012493
            1          1          why           19  ...  498.685784  31.192237  404.663563
            2          1    interview           25  ...  389.294931  19.083146  209.778063
            3          6         step           15  ...   26.082417  34.739663  607.977022
            <BLANKLINE>
            [4 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

            Finally, you can mark the origin of your datasets in dedicated columns in
            the resulting dataset's dataframes.

            >>> example1 = dummy_dataset(
            ...     2, 2, seed=0, label_map={0: "car"}, dataset_name="A"
            ... )
            >>> example2 = dummy_dataset(
            ...     2, 2, seed=1, label_map={0: "car"}, dataset_name="B"
            ... )
            >>> example1
            Dataset object containing 2 images and 2 objects
            Name :
                A
            Images root :
                such/serious
            Images :
                width  height relative_path   type  split
            id
            0     865     560  step/why.jpg   .jpg  train
            1     673     342  help/me.jpeg  .jpeg    val
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0          car            0  ...  511.143123  616.718121   12.497434
            1          0          car            0  ...  339.716034  233.243139  117.161956
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {0: 'car'}
            >>> example2
            Dataset object containing 2 images and 2 objects
            Name :
                B
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     525     779   reach/marriage.jpg  .jpg  train
            1     560     955  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0          car            0  ...   21.468549  283.211413  308.302755
            1          0          car            0  ...  586.986712  124.825174   57.793609
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {0: 'car'}
            >>> merged_examples = example1.merge(
            ...     example2, mark_origin=True, ignore_index=True
            ... )
            >>> merged_examples
            Dataset object containing 4 images and 4 objects
            Name :
                A+B
            Images root :
                .
            Images :
                width  height                     relative_path  ...  split origin origin_id
            id                                                   ...
            0     673     342         such/serious/help/me.jpeg  ...    val      A         1
            1     865     560         such/serious/step/why.jpg  ...  train      A         0
            2     560     955  care/suggest/determine/story.jpg  ...  train      B         1
            3     525     779   care/suggest/reach/marriage.jpg  ...  train      B         0
            <BLANKLINE>
            [4 rows x 7 columns]
            Annotations :
                image_id category_str  category_id  ...  box_height  origin  origin_id
            id                                      ...
            0          1          car            0  ...   12.497434       A          0
            1          1          car            0  ...  117.161956       A          1
            2          3          car            0  ...   57.793609       B          1
            3          3          car            0  ...  308.302755       B          0
            <BLANKLINE>
            [4 rows x 10 columns]
            Label map :
            {0: 'car'}

            By default, dataset which already feature an origin for its sample will
            retain it for further merges. Optionally, you can decide to overwrite the
            origin to the actual dataset that is being merged and forget the old origin.

            >>> example3 = dummy_dataset(
            ...     2, 2, seed=2, label_map={0: "car"}, dataset_name="C"
            ... )
            >>> merged_examples.merge(example3, mark_origin=True, ignore_index=True)
            Dataset object containing 6 images and 6 objects
            Name :
                A+B+C
            Images root :
                .
            Images :
                width  height                     relative_path  ...  split origin origin_id
            id                                                   ...
            0     560     955  care/suggest/determine/story.jpg  ...  train      B         1
            1     525     779   care/suggest/reach/marriage.jpg  ...  train      B         0
            2     673     342         such/serious/help/me.jpeg  ...    val      A         1
            3     865     560         such/serious/step/why.jpg  ...  train      A         0
            4     335     368        what/way/police/enter.jpeg  ...  train      C         1
            5     853     198  what/way/relationship/table.tiff  ...  train      C         0
            <BLANKLINE>
            [6 rows x 7 columns]
            Annotations :
                image_id category_str  category_id  ...  box_height  origin  origin_id
            id                                      ...
            0          1          car            0  ...   57.793609       B          1
            1          1          car            0  ...  308.302755       B          0
            2          3          car            0  ...   12.497434       A          0
            3          3          car            0  ...  117.161956       A          1
            4          4          car            0  ...  137.766169       C          1
            5          5          car            0  ...   14.083247       C          0
            <BLANKLINE>
            [6 rows x 10 columns]
            Label map :
            {0: 'car'}
            >>> merged_examples.merge(
            ...     example3, mark_origin=True, ignore_index=True, overwrite_origin=True
            ... )
            Dataset object containing 6 images and 6 objects
            Name :
                A+B+C
            Images root :
                .
            Images :
                width  height                     relative_path  ...  split origin origin_id
            id                                                   ...
            0     560     955  care/suggest/determine/story.jpg  ...  train    A+B         2
            1     525     779   care/suggest/reach/marriage.jpg  ...  train    A+B         3
            2     673     342         such/serious/help/me.jpeg  ...    val    A+B         0
            3     865     560         such/serious/step/why.jpg  ...  train    A+B         1
            4     335     368        what/way/police/enter.jpeg  ...  train      C         1
            5     853     198  what/way/relationship/table.tiff  ...  train      C         0
            <BLANKLINE>
            [6 rows x 7 columns]
            Annotations :
                image_id category_str  category_id  ...  box_height  origin  origin_id
            id                                      ...
            0          1          car            0  ...   57.793609     A+B          2
            1          1          car            0  ...  308.302755     A+B          3
            2          3          car            0  ...   12.497434     A+B          0
            3          3          car            0  ...  117.161956     A+B          1
            4          4          car            0  ...  137.766169       C          1
            5          5          car            0  ...   14.083247       C          0
            <BLANKLINE>
            [6 rows x 10 columns]
            Label map :
            {0: 'car'}

        """
        from .merge import merge_datasets

        return merge_datasets(
            self,
            other,
            allow_overlapping_image_ids=allow_overlapping_image_ids,
            realign_label_map=realign_label_map,
            ignore_index=ignore_index,
            mark_origin=mark_origin,
            overwrite_origin=overwrite_origin,
        )

    def __radd__(self, other: "int | Dataset") -> "Dataset":
        if isinstance(other, int):
            # Stub function so that we can use the sum function for datasets
            return self
        else:
            return self.__add__(other)

    def __add__(self, other: "Dataset") -> "Dataset":
        """Overloading of the "+" operator for Datasets.

        It will call the :meth:`Dataset.merge` method multiple times if needed:

        - Once with default parameters
        - If it fails because of an incompatible label map, it will try to remap the
          other dataset's label map to match this dataset.
        - If it fails because of another error, it will try to use the ``ignore_index``
          option set to ``True``.

        See Also:
            :meth:`Dataset.merge`

        Args:
            other: Other dataset to merge the first dataset with.

        Returns:
            Merged Dataset.
        """
        try:
            return self.merge(other)
        except IncompatibleLabelMapsError:
            warn(
                "Addition failed because of incompatible label maps, trying to"
                " remap classes of right value and retry the merge",
                RuntimeWarning,
            )
            return self + other.remap_from_other(self)
        except (ValueError, AssertionError):
            warn(
                "Addition failed, retrying merge with ignore_index set to True",
                RuntimeWarning,
            )
            return self.merge(other, ignore_index=True)

    def __sub__(self, other: "Dataset") -> tuple["Dataset", "Dataset", "Dataset"]:
        from lours.utils.difftools import dataset_diff

        assert isinstance(other, Dataset), "subtracted object must be a dataset"
        return dataset_diff(left_dataset=self, right_dataset=other)

    def remove_empty_images(self) -> Self:
        """Remove images without annotations from dataset.

        Note: This does NOT remove empty images from the disk, but simply from the
        dataset object, and thus they will not be copied when saving the dataset
        elsewhere.

        Returns:
            Dataset object with the rows of empty images removed from the
            ``self.images`` dataframe.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 1, seed=0)
            >>> example
            Dataset object containing 2 images and 1 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type split
            id
            0     342     136       help/me.jpeg  .jpeg  eval
            1     377     167  whatever/wait.png   .png   val
            Annotations :
                image_id category_str  category_id  ...  box_y_min  box_width  box_height
            id                                      ...
            0          0    interview           25  ...  73.932999  62.674584    8.569467
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example.remove_empty_images()
            Dataset object containing 1 image and 1 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height relative_path   type split
            id
            0     342     136  help/me.jpeg  .jpeg  eval
            Annotations :
                image_id category_str  category_id  ...  box_y_min  box_width  box_height
            id                                      ...
            0          0    interview           25  ...  73.932999  62.674584    8.569467
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

        """
        not_empty = self.images.index.isin(self.annotations["image_id"])
        return self.loc[not_empty]

    def cap_bounding_box_coordinates(self) -> Self:
        """Method to ensure the bounding box coordinates are inside the picture frame.
        Indeed, some dataset (like crowdhuman) do use outside of picture bounding box

        Returns:
            New Dataset with bounding box capped so that X and Y coordinates are inside
            corresponding picture dimensions

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> from lours.utils.testing import assert_bounding_boxes_well_formed
            >>> example = dummy_dataset(1, 1)
            >>> example.annotations.loc[0, "box_y_min"] = -0.5
            >>> example.annotations.loc[0, "box_height"] = (
            ...     example.images["height"][0] + 1
            ... )
            >>> example
            Dataset object containing 1 image and 1 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height relative_path   type  split
            id
            0     342     377  help/me.jpeg  .jpeg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            0          0    interview           25  ...      -0.5  306.509956       378.0
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

            >>> assert_bounding_boxes_well_formed(example)
            Traceback (most recent call last):
                ...
            AssertionError: Assertion failed. Bounding boxes must have positive Y values. First occurrence at row 0 : image_id                0
            category_str     interview
            category_id             25
            split                train
            box_x_min         5.652451
            box_y_min             -0.5
            box_width       306.509956
            box_height           378.0
            Name: 0, dtype: object

            >>> example.cap_bounding_box_coordinates()
            Dataset object containing 1 image and 1 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height relative_path   type  split
            id
            0     342     377  help/me.jpeg  .jpeg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            0          0    interview           25  ...       0.0  306.509956       377.0
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}

        """
        xmin, ymin, box_width, box_height = BBOX_COLUMN_NAMES
        im_dimensions = self.images.loc[
            self.annotations["image_id"], ["width", "height"]
        ]
        im_dimensions.index = self.annotations.index
        capped_xmin_series = self.annotations[xmin].clip(0, im_dimensions["width"])
        capped_ymin_series = self.annotations[ymin].clip(0, im_dimensions["height"])
        capped_box_width_series = self.annotations[box_width].clip(
            0, im_dimensions["width"] - capped_xmin_series
        )
        capped_box_height_series = self.annotations[box_height].clip(
            0, im_dimensions["height"] - capped_ymin_series
        )
        capped_annotations = self.annotations.assign(
            **{
                xmin: capped_xmin_series,
                ymin: capped_ymin_series,
                box_width: capped_box_width_series,
                box_height: capped_box_height_series,
            }
        )
        return self.from_template(annotations=capped_annotations)

    def booleanize(
        self,
        column_names: str | Iterable[str] | None = None,
        missing_ok: bool = False,
        **possible_values: set,
    ) -> Self:
        """Convert given column in ``self.images`` or ``self.annotations`` from lists to
        columns of booleans.

        See :func:`.util.column_booleanize.booleanize`

        Note:
            in the case column name is present in both images and annotations, the
            column in ``self.images`` takes precedence

        Args:
            column_names: columns to convert. After conversion, it will be dropped
                from corresponding DataFrames
            missing_ok: If set to True, will not raise a KeyError if the column name is
                neither in ``self.images`` nor ``self.annotations``
            **possible_values: keyword arguments dictionary for possible values. If a
                column name in ``column_names`` is not present in this dictionary, will
                deduce from occurrence in the dataset

        Raises:
            KeyError: if ``missing_ok`` is set to False, the given ``column_name`` must
                be either in ``self.images`` columns or in ``self.annotations`` columns.
            TypeError: When for a particular column possible values need to be deduced,
                the column must have value that are all iterable except strings.

        Returns:
            New dataset with multiple boolean columns in the form
            ``{column_name}.{value}``.

        See Also:
            :ref:`related tutorial </notebooks/7_demo_booleanize.ipynb>`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(
            ...     n_imgs=3,
            ...     n_annot=3,
            ...     n_list_columns_images=[2, 3],
            ...     n_list_columns_annotations=1,
            ... )
            >>> example
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ...                         beyond                father
            id                 ...
            0     342     167  ...                       [enough]  [challenge, someone]
            1     377     114  ...          [present, successful]           [challenge]
            2     136     257  ...  [present, successful, enough]  [challenge, someone]
            <BLANKLINE>
            [3 rows x 7 columns]
            Annotations :
                image_id category_str  ...  box_height                                   where
            id                         ...
            0          2          why  ...  138.451739  [no, season, play, choice, force, bit]
            1          1          why  ...   63.576932                     [no, choice, force]
            2          2         step  ...   99.999123           [no, season, play, week, bit]
            <BLANKLINE>
            [3 rows x 9 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> modified = example.booleanize(column_names=["beyond", "where"])
            >>> modified
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ... beyond.present beyond.successful
            id                 ...
            0     342     167  ...          False             False
            1     377     114  ...           True              True
            2     136     257  ...           True              True
            <BLANKLINE>
            [3 rows x 9 columns]
            Annotations :
                image_id category_str  category_id  ... where.play  where.season  where.week
            id                                      ...
            0          2          why           19  ...       True          True       False
            1          1          why           19  ...      False         False       False
            2          2         step           15  ...       True          True        True
            <BLANKLINE>
            [3 rows x 15 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> modified.annotations.dtypes
            image_id          int64
            category_str     object
            category_id       int64
            split            object
            box_x_min       float64
            box_y_min       float64
            box_width       float64
            box_height      float64
            where.bit          bool
            where.choice       bool
            where.force        bool
            where.no           bool
            where.play         bool
            where.season       bool
            where.week         bool
            dtype: object
            >>> modified.booleanized_columns
            {'images': {'beyond'}, 'annotations': {'where'}}

            >>> example.booleanize(beyond={"enough", "successful"})
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ... beyond.enough beyond.successful
            id                 ...
            0     342     167  ...          True             False
            1     377     114  ...         False              True
            2     136     257  ...          True              True
            <BLANKLINE>
            [3 rows x 8 columns]
            Annotations :
                image_id category_str  ...  box_height                                   where
            id                         ...
            0          2          why  ...  138.451739  [no, season, play, choice, force, bit]
            1          1          why  ...   63.576932                     [no, choice, force]
            2          2         step  ...   99.999123           [no, season, play, week, bit]
            <BLANKLINE>
            [3 rows x 9 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
        """
        if column_names is None:
            column_names = set()
        if isinstance(column_names, str):
            column_names = {column_names}
        else:
            column_names = {*column_names}
        if possible_values:
            column_names = set(column_names).union(possible_values.keys())
        elif not column_names:
            # Nothing to booleanize, return immediately
            warn("Nothing to booleanize, dataset returned as is", RuntimeWarning)
            return self

        images_booleanize = set()
        annotations_booleanize = set()
        while column_names:
            name = column_names.pop()
            if name in self.images.columns:
                images_booleanize.add(name)
            elif name in self.annotations.columns:
                annotations_booleanize.add(name)
            elif not missing_ok:
                raise KeyError(
                    f"Column name {name} is neither in self.images nor self.annotations"
                )
        new_images = booleanize(
            self.images,
            separator=".",
            **{name: possible_values.get(name, None) for name in images_booleanize},
        )
        new_annotations = booleanize(
            self.annotations,
            separator=".",
            **{
                name: possible_values.get(name, None) for name in annotations_booleanize
            },
        )
        output_dataset = self.from_template(
            images=new_images,
            annotations=new_annotations,
            reset_booleanized=False,
        )
        output_dataset.booleanized_columns["images"] |= set(images_booleanize)
        output_dataset.booleanized_columns["annotations"] |= set(annotations_booleanize)
        return output_dataset

    def debooleanize(
        self,
        dataframe: Literal["both", "images", "annotations"] = "both",
    ) -> Self:
        """Convert booleanized columns back to list form, for exporting purpose.

        Note:
            This will only debooleanize columns that have been explicitly booleanized,
            and not just boolean columns. It will look for values in
            ``self.booleanized_columns`` and retrieve all the column with the name
            in the form ``column_name.entry`` to reconstruct the ``column_name``
            column.

        Args:
            dataframe: Which dataframe you want to booleanize.
                Can be either "images", "annotations" or None.
                If set to None, will debooleanize both dataframes. Defaults to None.

        Returns:
            New dataset object with converted columns, booleanized columns are dropped.

        See Also:
            :ref:`related tutorial </notebooks/7_demo_booleanize.ipynb>`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(
            ...     n_imgs=3,
            ...     n_annot=3,
            ...     n_list_columns_images=[2, 3],
            ...     n_list_columns_annotations=1,
            ... )
            >>> example
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ...                         beyond                father
            id                 ...
            0     342     167  ...                       [enough]  [challenge, someone]
            1     377     114  ...          [present, successful]           [challenge]
            2     136     257  ...  [present, successful, enough]  [challenge, someone]
            <BLANKLINE>
            [3 rows x 7 columns]
            Annotations :
                image_id category_str  ...  box_height                                   where
            id                         ...
            0          2          why  ...  138.451739  [no, season, play, choice, force, bit]
            1          1          why  ...   63.576932                     [no, choice, force]
            2          2         step  ...   99.999123           [no, season, play, week, bit]
            <BLANKLINE>
            [3 rows x 9 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> modified = example.booleanize(column_names=["beyond", "where"])
            >>> modified
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ... beyond.present beyond.successful
            id                 ...
            0     342     167  ...          False             False
            1     377     114  ...           True              True
            2     136     257  ...           True              True
            <BLANKLINE>
            [3 rows x 9 columns]
            Annotations :
                image_id category_str  category_id  ... where.play  where.season  where.week
            id                                      ...
            0          2          why           19  ...       True          True       False
            1          1          why           19  ...      False         False       False
            2          2         step           15  ...       True          True        True
            <BLANKLINE>
            [3 rows x 15 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> modified.debooleanize()
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ...                         beyond                father
            id                 ...
            0     342     167  ...                       [enough]  [challenge, someone]
            1     377     114  ...          [present, successful]           [challenge]
            2     136     257  ...  [enough, present, successful]  [challenge, someone]
            <BLANKLINE>
            [3 rows x 7 columns]
            Annotations :
                image_id category_str  ...  box_height                                   where
            id                         ...
            0          2          why  ...  138.451739  [bit, choice, force, no, play, season]
            1          1          why  ...   63.576932                     [choice, force, no]
            2          2         step  ...   99.999123           [bit, no, play, season, week]
            <BLANKLINE>
            [3 rows x 9 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> modified.debooleanize(dataframe="images")
            Dataset object containing 3 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height  ...                         beyond                father
            id                 ...
            0     342     167  ...                       [enough]  [challenge, someone]
            1     377     114  ...          [present, successful]           [challenge]
            2     136     257  ...  [enough, present, successful]  [challenge, someone]
            <BLANKLINE>
            [3 rows x 7 columns]
            Annotations :
                image_id category_str  category_id  ... where.play  where.season  where.week
            id                                      ...
            0          2          why           19  ...       True          True       False
            1          1          why           19  ...      False         False       False
            2          2         step           15  ...       True          True        True
            <BLANKLINE>
            [3 rows x 15 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
        """
        images = self.images
        annotations = self.annotations
        do_images = dataframe in ["both", "images"]
        do_annot = dataframe in ["both", "annotations"]
        if do_images:
            images = debooleanize(images, self.booleanized_columns["images"], ".")
        if do_annot:
            annotations = debooleanize(
                annotations, self.booleanized_columns["annotations"], "."
            )
        output = self.from_template(
            images=images, annotations=annotations, reset_booleanized=True
        )
        return output

    def remap_classes(
        self,
        class_mapping: dict[int, int],
        new_names: dict[int, str] | None = None,
        remove_not_mapped: bool = True,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Remap classes ids and names according to a dictionary

        Note:
            In case of class fusion, the class name of the last category_id with respect
            to ``class_mapping`` order will be deduced.

        Note:
            if ``remove_not_mapped`` is True, Classes that are not present in the
            dictionary are removed from the dataset altogether.
            Otherwise, they are kept as if the identity mapping was in the bottom of
            ``class_mapping`` for this particular class.
            For potential class fusion, the name of the unmapped class will be used.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_csv`
            - :meth:`.remap_from_other`
            - :meth:`.remove_classes`
            - :meth:`.keep_classes`

        Args:
            class_mapping: ``old_id`` -> ``new_id`` mapping
            new_names: Optimal ``new_id`` -> ``new_name`` mapping, essentially the new
                label_map.
                If category_id is missing from keys, will deduce it from the former one.
                Defaults to None.
            remove_not_mapped: If set to True, will remove classes that are not in
                class mapping. Otherwise, keep them as is (with potential class fusion).
                Defaults to True.
            remove_emptied_images: If set to True, will remove from self.images the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            New dataset object with updated label maps, category ids and
            category_names

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.remap_classes({14: 1})
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen            1  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {1: 'listen'}

            >>> example.remap_classes({14: 1}, remove_not_mapped=False)
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen            1  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {1: 'listen', 15: 'marriage', 22: 'reach'}

            >>> example.remap_classes(
            ...     {14: 1},
            ...     remove_not_mapped=False,
            ...     new_names={1: "new_listen", 15: "new_marriage"},
            ... )
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1   new_listen            1  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {1: 'new_listen', 15: 'new_marriage', 22: 'reach'}

            >>> example.remap_classes(
            ...     {14: 1},
            ...     remove_emptied_images=True,
            ... )
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height   relative_path  type  split
            id
            1     131     840  air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min  box_width  box_height
            id                                      ...
            0          1       listen            1  ...  276.974642   9.718823  184.684056
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {1: 'listen'}

            Note that only empited images are removed. Images that were already empty
            before are kept

            >>> example = dummy_dataset(2, 2)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...  73.932999   71.552480   42.673983
            1          0          why           19  ...   4.567638  248.551257  122.602211
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example.remap_classes({25: 1}, remove_emptied_images=True)
            Dataset object containing 1 image and 0 object
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path  type  split
            id
            1     377     167  whatever/wait.png  .png  train
            Annotations :
            Empty DataFrame
            Columns: [image_id, category_str, category_id, split, box_x_min, box_y_min, box_width, box_height]
            Index: []
            Label map :
            {1: 'interview'}
        """  # noqa: E501
        if not remove_not_mapped:
            not_mapped = {
                category_id: category_id
                for category_id in self.label_map.keys()
                if category_id not in class_mapping.keys()
            }
            class_mapping = {**class_mapping, **not_mapped}
        new_label_map = {
            v: self.label_map[k]
            for k, v in class_mapping.items()
            if k in self.label_map
        }
        if new_names is not None:
            new_label_map = {**new_label_map, **new_names}
        # Only keep classes referenced in the class_mapping
        new_annotations = self.annotations[
            self.annotations["category_id"].isin(class_mapping)
        ]
        # Replace both id and class in annotation dataframe
        # Note that the ignore is probably caused by a bug.
        # See https://github.com/pandas-dev/pandas-stubs/issues/1161
        # See https://github.com/microsoft/pyright/issues/10057
        new_annotations = new_annotations.replace(
            {"category_id": class_mapping}  # pyright: ignore
        )
        new_annotations["category_str"] = new_annotations["category_id"].map(
            new_label_map
        )
        if remove_emptied_images:
            already_empty_images = ~self.images.index.isin(self.annotations["image_id"])
            already_empty_images_ids = self.images.index[already_empty_images].tolist()
            remaining_images = new_annotations["image_id"].unique().tolist()
            new_images = self.images.loc[[*already_empty_images_ids, *remaining_images]]
        else:
            new_images = self.images

        return self.from_template(
            images=new_images,
            annotations=new_annotations,
            label_map=new_label_map,
        )

    def remap_from_preset(
        self,
        input_dataset_map: str,
        output_dataset_map: str,
        remove_not_mapped: bool = True,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Same as class remap, but instead of taking a dictionary, you give the name
        of a preset. Registered presets are stored in remap_presets folders, with csv
        files in the form ``{inputr_dataset_map}_to_{output_dataset_map}``

        Args:
            input_dataset_map: Name of label map to convert from.
            output_dataset_map: Name of label to convert to.
            remove_not_mapped: If set to True, will remove classes that are not in
                class mapping. Otherwise, keep them as is (with potential class fusion).
                Defaults to True.
            remove_emptied_images: If set to True, will remove from ``self.images`` the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            New dataset object with remapped classes according to the preset

        Raises:
            KeyError: raised when the input/output pair does not exists in presets.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_csv`
            - :meth:`.remap_from_other`
            - :meth:`.remove_classes`
            - :meth:`.keep_classes`
        """
        from . import remap_presets

        try:
            id_mapping, new_names = remap_presets.presets[
                (input_dataset_map, output_dataset_map)
            ]
        except KeyError as e:
            raise ValueError(
                "Preset not available. Available presets are : \n"
                f"{remap_presets.list_available_presets()}"
            ) from e
        return self.remap_classes(
            id_mapping, new_names, remove_not_mapped, remove_emptied_images
        )

    def remap_from_dataframe(
        self,
        df: pd.DataFrame,
        remove_not_mapped: bool = True,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Same as class remap, but instead of taking a dictionary, you give a
        dataframe.

        Dataframe must have at least these two columns:

         - ``input_category_id``
         - ``output_category_id``

        Optional columns for category names:

         - ``input_category_name``
         - ``output_category_name``

        Args:
            df: dataframe with aforementioned columns
            remove_not_mapped: If set to True, will remove classes that are not in
                class mapping. Otherwise, keep them as is (with potential class fusion).
                Defaults to True.
            remove_emptied_images: If set to True, will remove from self.images the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            new dataset object with remapped classes according to
            the given table in the dataframe

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_csv`
            - :meth:`.remap_from_other`
            - :meth:`.remove_classes`
            - :meth:`.keep_classes`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> remap_df = pd.DataFrame(
            ...     data={
            ...         "input_category_id": [14, 22],
            ...         "output_category_id": [0, 1],
            ...         "output_category_name": ["new_listen", "new_reach"],
            ...     }
            ... )
            >>> remap_df
               input_category_id  output_category_id output_category_name
            0                 14                   0           new_listen
            1                 22                   1            new_reach
            >>> example.remap_from_dataframe(remap_df)
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1   new_listen            0  ...  276.974642    9.718823  184.684056
            1          0    new_reach            1  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {0: 'new_listen', 1: 'new_reach'}
        """  # noqa: E501
        if df.index.name == "input_category_id":
            mapping_df = df
        else:
            mapping_df = df.set_index("input_category_id")
        mapping_dict = mapping_df["output_category_id"].to_dict()
        mapping_names = (
            mapping_df.groupby("output_category_id")["output_category_name"]
            .first()
            .to_dict()
        )
        return self.remap_classes(
            mapping_dict,  # pyright: ignore
            mapping_names,  # pyright: ignore
            remove_not_mapped,
            remove_emptied_images,
        )

    def remap_from_csv(
        self,
        csv: Path,
        remove_not_mapped: bool = True,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Same as class remap, but instead of taking a dictionary, you give the path
        to a csv file.

        csv file must have at least these two columns :

         - ``input_category_id``
         - ``output_category_id``

        Optional columns for category names :

         - ``input_category_name``
         - ``output_category_name``

        Args:
            csv: path to csv file, to be read by pandas
            remove_not_mapped: If set to True, will remove classes that are not in
                class mapping. Otherwise, keep them as is (with potential class fusion).
                Defaults to True.
            remove_emptied_images: If set to True, will remove from self.images the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            New dataset object with remapped classes according to
            the given table in the csv file

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_other`
            - :meth:`.remove_classes`
            - :meth:`.keep_classes`
        """  # noqa: E501
        mapping_df = pd.read_csv(csv).set_index("input_category_id")
        return self.remap_from_dataframe(
            mapping_df, remove_not_mapped, remove_emptied_images
        )

    def remap_from_other(
        self,
        other: "Dataset",
        remove_not_mapped: bool = False,
        remove_emptied_images: bool = False,
    ) -> Self:
        """Try to remap classes of dataset to match the ones in another dataset by
        retrieving categories with the same name.

        This is useful when trying to merge together two dataset with incompatible label
        maps.

        The mapping is constructed so that no category id represents different category
        labels between other dataset and remapped dataset.

        This function works by first applying the mapping on objects with the same
        category strings as some other objects in other dataset, and reassign the other
        categories so that the ids don't overlap. categories whose name is only present
        in the current and have the same id as some other category in the other dataset
        will be iteratively set to the lowest unoccupied category id of all label maps.

        Note:
            The name of a category is ambiguous. Another method of class remapping
            should be preferred if possible.

        See :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`

        Args:
            other: Other dataset to align the output's label map with.
            remove_not_mapped: If set to True, will remove classes that are in self,
                but not in other dataset's class mapping.
                Otherwise, keep them as is. Defaults to False.
            remove_emptied_images: If set to True, will remove from self.images the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Raises:
            AssertionError: Error raised if label map of one of the two dataset don't
                have unique category names.

        Returns:
            Dataset: New dataset with remapped classes to match the ones in `other`

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_csv`
            - :meth:`.remove_classes`
            - :meth:`.keep_classes`

        Example:
            current dataset has label map ``{1: car, 2: person, 3:truck}`` and other
            dataset has label map ``{1: train, 2: car, 3: person}``. This method will
            construct this mapping dictionary: ``{1: 2, 2: 3, 3: 4}`` so that the
            remapped dataset has the following label map: ``{2:car, 3:person, 4:truck}``
            which is now compatible with other dataset's label map (no overlap)

            In the case you merge the two datasets, the resulting merged label map will
            be: ``{1: train, 2: car, 3: person, 4: truck}``

            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example1 = dummy_dataset(
            ...     n_imgs=2,
            ...     n_annot=2,
            ...     label_map={1: "car", 2: "person", 3: "truck"},
            ...     seed=3,
            ... )
            >>> example1
            Dataset object containing 2 images and 2 objects
            Name :
                have_page_personal
            Images root :
                draw/name
            Images :
                width  height   relative_path  type  split
            id
            0     830     261  add/police.bmp  .bmp  train
            1     177     313    ok/event.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0        truck            3  ...  102.110558  531.572263   22.921831
            1          1       person            2  ...   49.998280   56.543521  111.741397
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {1: 'car', 2: 'person', 3: 'truck'}
            >>> example2 = dummy_dataset(
            ...     n_imgs=2,
            ...     n_annot=2,
            ...     label_map={1: "train", 2: "car", 3: "person"},
            ...     seed=1,
            ... )
            >>> example2
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     525     779   reach/marriage.jpg  .jpg  train
            1     560     955  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0       person            3  ...  586.986712  124.825174   57.793609
            1          0       person            3  ...  318.766127  207.777851  100.447514
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {1: 'train', 2: 'car', 3: 'person'}
            >>> example1.remap_from_other(example2)
            Using the following class remapping dictionary :
            {1: 2, 2: 3, 3: 4}
            Dataset object containing 2 images and 2 objects
            Name :
                have_page_personal
            Images root :
                draw/name
            Images :
                width  height   relative_path  type  split
            id
            0     830     261  add/police.bmp  .bmp  train
            1     177     313    ok/event.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          0        truck            4  ...  102.110558  531.572263   22.921831
            1          1       person            3  ...   49.998280   56.543521  111.741397
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {2: 'car', 3: 'person', 4: 'truck'}
            >>> example1.remap_from_other(example2, remove_not_mapped=True)
            Using the following class remapping dictionary :
            {1: 2, 2: 3}
            Dataset object containing 2 images and 1 object
            Name :
                have_page_personal
            Images root :
                draw/name
            Images :
                width  height   relative_path  type  split
            id
            0     830     261  add/police.bmp  .bmp  train
            1     177     313    ok/event.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min  box_width  box_height
            id                                      ...
            1          1       person            3  ...  49.99828  56.543521  111.741397
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {2: 'car', 3: 'person'}
            >>> example1.remap_from_other(
            ...     example2, remove_not_mapped=True, remove_emptied_images=True
            ... )
            Using the following class remapping dictionary :
            {1: 2, 2: 3}
            Dataset object containing 1 image and 1 object
            Name :
                have_page_personal
            Images root :
                draw/name
            Images :
                width  height relative_path  type  split
            id
            1     177     313  ok/event.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min  box_width  box_height
            id                                      ...
            1          1       person            3  ...  49.99828  56.543521  111.741397
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {2: 'car', 3: 'person'}

        """
        from ..utils.testing import assert_label_map_well_formed

        def lowest_missing_value(input_list: Iterable[int]) -> int:
            sorted_values = sorted(set(input_list))
            for s1, s2 in zip(sorted_values[:-1], sorted_values[1:]):
                if s2 - s1 > 1:
                    return s1 + 1
            return max(sorted_values) + 1

        assert_label_map_well_formed(self)
        assert_label_map_well_formed(other)
        class_mapping = {}
        inverted_label_map_reference = {v: k for k, v in other.label_map.items()}
        for k, v in self.label_map.items():
            new_id = inverted_label_map_reference.get(v)
            if new_id is not None:
                class_mapping[k] = new_id
            elif not remove_not_mapped:
                if k in other.label_map.keys():
                    class_mapping[k] = lowest_missing_value(
                        [
                            *self.label_map,
                            *other.label_map,
                            *class_mapping.values(),
                        ]
                    )
                else:
                    # This is not needed, but the printed remapping dictionary will be
                    # more comprehensive that way
                    class_mapping[k] = k
        print(
            "Using the following class remapping dictionary"
            f" :\n{pformat(class_mapping)}"
        )
        return self.remap_classes(
            class_mapping=class_mapping,
            remove_not_mapped=remove_not_mapped,
            remove_emptied_images=remove_emptied_images,
        )

    def remove_classes(
        self, to_remove: int | Iterable[int], remove_emptied_images: bool = False
    ) -> Self:
        """Perform a simple remapping, where given classes are removed

        Notes:
            - This function is equivalent to calling :meth:`.remap_classes` where the
              remapping dictionary is the identity except removed classes do not appear.
            - This function is the complementary to :meth:`.keep_classes`.

        Args:
            to_remove: list of class ids to remove.
            remove_emptied_images: If set to True, will remove from ``self.images`` the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            New dataset object where given classes have been removed

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.keep_classes`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_csv`
            - :meth:`.remap_from_other`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.remove_classes(14)
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'marriage', 22: 'reach'}

            >>> example.remove_classes([14, 15])
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {22: 'reach'}

            >>> example.remove_classes(14, remove_emptied_images=True)
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'marriage', 22: 'reach'}
        """
        if isinstance(to_remove, int):
            to_remove = [to_remove]
        class_mapping = {i: i for i in self.label_map if i not in to_remove}
        return self.remap_classes(
            class_mapping,
            remove_not_mapped=True,
            remove_emptied_images=remove_emptied_images,
        )

    def keep_classes(
        self, to_keep: int | Iterable[int], remove_emptied_images: bool = False
    ) -> Self:
        """Perform a simple remapping, where given classes kept, and other are removed

        Notes:
            - This function is equivalent to calling :meth:`.remap_classes` where the
              remapping dictionary is the identity except only kept classes appear.
            - This function is the complementary to :meth:`.remove_classes`.

        Args:
            to_keep: list of class ids to keep.
            remove_emptied_images: If set to True, will remove from ``self.images`` the
                images that are now empty of annotation.
                Note that it will keep the images that were empty before the remapping.
                Defaults to False.

        Returns:
            New dataset object where given classes have been kept, and the rest removed.

        See Also:
            - :ref:`related tutorial </notebooks/1_demo_dataset.ipynb#Remap-classes>`
            - :meth:`.remap_classes`
            - :meth:`.remap_from_preset`
            - :meth:`.remap_from_dataframe`
            - :meth:`.remap_from_csv`
            - :meth:`.remap_from_other`
            - :meth:`.remove_classes`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(2, 2, seed=1)
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0          1       listen           14  ...  276.974642    9.718823  184.684056
            1          0        reach           22  ...    6.311037  123.141689  174.239136
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.keep_classes([15, 22])
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'marriage', 22: 'reach'}

            >>> example.keep_classes(22)
            Dataset object containing 2 images and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            1     131     840       air/method.bmp  .bmp  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {22: 'reach'}

            >>> example.keep_classes([15, 22], remove_emptied_images=True)
            Dataset object containing 1 image and 1 object
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path  type  split
            id
            0     955     229  determine/story.jpg  .jpg  train
            Annotations :
                image_id category_str  category_id  ... box_y_min   box_width  box_height
            id                                      ...
            1          0        reach           22  ...  6.311037  123.141689  174.239136
            <BLANKLINE>
            [1 rows x 8 columns]
            Label map :
            {15: 'marriage', 22: 'reach'}
        """
        if isinstance(to_keep, int):
            to_keep = [to_keep]
        class_mapping = {i: i for i in self.label_map if i in to_keep}
        return self.remap_classes(
            class_mapping,
            remove_not_mapped=True,
            remove_emptied_images=remove_emptied_images,
        )

    def simple_split(
        self,
        input_seed: int = 0,
        split_names: Sequence[str] = ("train", "valid"),
        target_split_shares: Sequence[float] = (0.8, 0.2),
        inplace: bool = False,
    ) -> Self:
        """Simple version of splitting method, splitting images randomly.

        Args:
            input_seed: Random seed for splitting images. Defaults to 0.
            split_names: Names of splits. Must be more than 1 element long and the same
                size as ``target_split_shares``. Defaults to ``("train", "valid")``.
            target_split_shares: Share values of each split. Must be the same size as
                ``split_names``. Must add up to 1. Defaults to ``(0.8, 0.2)``.
            inplace: If set to True, will perform the splitting inplace without creating
                a new dataset. Defaults to False.

        Returns:
            Dataset with new splits applied to its images DataFrame.

        See Also:
            - More in-depth explanation in this :ref:`tutorial </notebooks/2_demo_split.ipynb>`
            - :meth:`split`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(200, 200, seed=1, split_names=None)
            >>> example
            Dataset object containing 200 images and 200 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path   type
            id
            0      955     488  determine/story.jpg   .jpg
            1      131     895       air/method.bmp   .bmp
            2      229     880   political/lead.jpg   .jpg
            3      840     384        like/safe.bmp   .bmp
            4      953     668      suffer/set.jpeg  .jpeg
            ..     ...     ...                  ...    ...
            195    122     437    state/almost.tiff  .tiff
            196    752     300     weight/tend.jpeg  .jpeg
            197    554     228  remember/summer.png   .png
            198    688     605       yet/though.png   .png
            199    243     227   describe/road.tiff  .tiff
            <BLANKLINE>
            [200 rows x 4 columns]
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          77     marriage           15  ...  425.688592   29.159255   39.517594
            1         137     marriage           15  ...  383.838546  551.353799  285.211136
            2         158     marriage           15  ...  174.889594  144.774339  183.531195
            3         111        reach           22  ...  151.265769   97.611967  282.485307
            4         121     marriage           15  ...   38.236459  522.170458   36.783181
            ..        ...          ...          ...  ...         ...         ...         ...
            195       129        reach           22  ...  190.935508  104.385252    3.669239
            196        33       listen           14  ...  322.704987  469.556266  193.375897
            197       181       listen           14  ...  403.794364  349.250089   66.745395
            198        55        reach           22  ...    2.534284  119.223978  110.346924
            199        89        reach           22  ...  172.664334  658.570932  282.920285
            <BLANKLINE>
            [200 rows x 7 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> splitted = example.simple_split()
            >>> splitted
            Dataset object containing 200 images and 200 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path   type  split
            id
            0      955     488  determine/story.jpg   .jpg  train
            1      131     895       air/method.bmp   .bmp  train
            2      229     880   political/lead.jpg   .jpg  train
            3      840     384        like/safe.bmp   .bmp  train
            4      953     668      suffer/set.jpeg  .jpeg  valid
            ..     ...     ...                  ...    ...    ...
            195    122     437    state/almost.tiff  .tiff  train
            196    752     300     weight/tend.jpeg  .jpeg  train
            197    554     228  remember/summer.png   .png  train
            198    688     605       yet/though.png   .png  valid
            199    243     227   describe/road.tiff  .tiff  train
            <BLANKLINE>
            [200 rows x 5 columns]
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                       ...
            0          77     marriage           15  ...  425.688592   29.159255   39.517594
            1         137     marriage           15  ...  383.838546  551.353799  285.211136
            2         158     marriage           15  ...  174.889594  144.774339  183.531195
            3         111        reach           22  ...  151.265769   97.611967  282.485307
            4         121     marriage           15  ...   38.236459  522.170458   36.783181
            ..        ...          ...          ...  ...         ...         ...         ...
            195       129        reach           22  ...  190.935508  104.385252    3.669239
            196        33       listen           14  ...  322.704987  469.556266  193.375897
            197       181       listen           14  ...  403.794364  349.250089   66.745395
            198        55        reach           22  ...    2.534284  119.223978  110.346924
            199        89        reach           22  ...  172.664334  658.570932  282.920285
            <BLANKLINE>
            [200 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> splitted.images["split"].value_counts() / len(splitted)
            split
            train    0.725
            valid    0.275
            Name: count, dtype: float64
        """
        if len(split_names) <= 1:
            raise ValueError(
                f"Must provide at least 2 split names. Got {split_names} of size"
                f" {len(split_names)} instead."
            )
        if len(target_split_shares) != len(split_names):
            raise ValueError(
                "Size mismatch between 'split_names' and 'split_shares'"
                f" ({len(split_names)} vs {len(target_split_shares)})"
            )
        if sum(target_split_shares) != 1:
            raise ValueError(
                "Split share values must addup to 1. Got"
                f" {sum(target_split_shares)} instead"
            )
        gen = np.random.default_rng(input_seed)
        split = gen.choice(
            list(split_names), size=len(self), p=list(target_split_shares)
        )
        if inplace:
            self.images["split"] = split
            return self
        else:
            return self.from_template(images=self.images.assign(split=split))

    def split(
        self,
        input_seed: int = 0,
        split_names: Sequence[str] = ("train", "valid"),
        target_split_shares: Sequence[float] = (0.8, 0.2),
        keep_separate_groups: group_list = ("image_id",),
        keep_balanced_groups: group_list = ("category_id",),
        keep_balanced_groups_weights: Sequence[float] | None = None,
        inplace: bool = False,
        hist_cost_weight: float = 1,
        share_cost_weight: float = 1,
        earth_mover_regularization: float = 0,
    ) -> Self:
        """Perform the split operation on annotations and images.

        This algorithm works in 2 steps:

        1. divide the dataframe into atomic sub frames. Given the image and annotation
            attributes that need to be kept separate, we can construct sub frame of
            elements that cannot be in different splits.
        2. Construct the split dataframes iteratively by trying to keep given column
            values with a balanced repartition between splits, along with keeping split
            sizes as close to target share as possible. Each atomic sub-dataframe is
            routed to the split that minimize a cost function which try to optimize
            repartition targets.

        Warning:
            if self.images and ``self.annotations`` each have a column with the same
            name, the column in ``self.images`` will be ignored. Make sure column names
            are mutually exclusive to avoid problems.

        See :func:`pandas.split_dataframe`

        Args:
            input_seed: Seed used for shuffling sub dataframes before beginning step 2
                of splitting algorithm. Defaults to 0.
            split_names: Names of splits. Must be the same length as ``split_shares``.
                Defaults to ("train", "valid").
            target_split_shares: List of target relative size of each split. Must be the
                same length as ``split_names``, and will be normalized so that its sum
                is 1. Defaults to (0.8, 0.2).
            keep_separate_groups: columns or groups
                (see :obj:`.group`) in annotations or images DataFrame to keep separate.
                That is for a particular column or group, two rows with the same value
                cannot be in different splits. Note that ``image_id`` will be added to
                that list, because split happen at the image level.
                Defaults to ("image_id",).
            keep_balanced_groups: columns or groups
                (see :obj:`.group`) in annotations or images DataFrame to keep balanced.
                That is for a particular group, the distribution of values is the same
                between original DataFrame and its split, as much as possible.
                Defaults to ("category_id",).
            keep_balanced_groups_weights: Importance of each group to keep balanced when
                computing histogram cost. If not None, must be a single float or the
                same size as ``keep_separate_groups``. Defaults to None.
            inplace: If set, will modify dataframes inplace. This can silently modify
                some objects (like Datasets) that use them but has a lower memory
                footprint. Defaults to False.
            hist_cost_weight: importance of histogram cost for balanced groups.
                The higher, the more important the histogram cost will be for the
                decision of where to put each split. Defaults to 1.
            share_cost_weight: importance of share cost for balanced groups.
                The higher, the more important the share cost will be for the decision
                of where to put each split. Defaults to 1.
            earth_mover_regularization: Regularization parameter applied to sinkhorn's
                algorithm during earth mover distance computation. See
                :func:`lours.dataset.split.balanced_group.earth_mover_distance`.
                Defaults to 0.

        Returns:
            new Dataset with the split column populated with the corresponding
            split names.

        See Also:
            - More in-depth explanation in this :ref:`tutorial </notebooks/2_demo_split.ipynb>`
            - :meth:`simple_split`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset(
            ...     200,
            ...     n_attribute_columns_images={"balanced": 10, "separate": 10},
            ...     split_names=None,
            ...     seed=1,
            ... )
            >>> example
            Dataset object containing 200 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path   type  balanced separate
            id
            0      955     488  determine/story.jpg   .jpg      send   system
            1      131     895       air/method.bmp   .bmp      note   system
            2      229     880   political/lead.jpg   .jpg  anything      law
            3      840     384        like/safe.bmp   .bmp  anything   likely
            4      953     668      suffer/set.jpeg  .jpeg  training   attack
            ..     ...     ...                  ...    ...       ...      ...
            195    122     437    state/almost.tiff  .tiff  anything     star
            196    752     300     weight/tend.jpeg  .jpeg     could     rest
            197    554     228  remember/summer.png   .png  anything   system
            198    688     605       yet/though.png   .png      note   number
            199    243     227   describe/road.tiff  .tiff       end   number
            <BLANKLINE>
            [200 rows x 6 columns]
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0         77        reach           22  ...   45.427512   40.116677  318.073851
            1        137     marriage           15  ...  202.481384  435.389400  475.375279
            <BLANKLINE>
            [2 rows x 7 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> example.images["separate"].value_counts()
            separate
            star      27
            likely    27
            number    27
            attack    22
            rest      20
            law       18
            entire    17
            enough    16
            system    15
            often     11
            Name: count, dtype: int64
            >>> splitted = example.split(
            ...     keep_balanced_groups=["balanced"], keep_separate_groups=["separate"]
            ... )
            Splitting annotations ...
            Separating input data into atomic chunks
            1 chunks to distribute across 2 splits
            Splitting images ...
            Separating input data into atomic chunks
            10 chunks to distribute across 2 splits
            >>> splitted
            Dataset object containing 200 images and 2 objects
            Name :
                shake_effort_many
            Images root :
                care/suggest
            Images :
                width  height        relative_path   type  split  balanced separate
            id
            0      955     488  determine/story.jpg   .jpg  train      send   system
            1      131     895       air/method.bmp   .bmp  train      note   system
            2      229     880   political/lead.jpg   .jpg  valid  anything      law
            3      840     384        like/safe.bmp   .bmp  train  anything   likely
            4      953     668      suffer/set.jpeg  .jpeg  train  training   attack
            ..     ...     ...                  ...    ...    ...       ...      ...
            195    122     437    state/almost.tiff  .tiff  valid  anything     star
            196    752     300     weight/tend.jpeg  .jpeg  train     could     rest
            197    554     228  remember/summer.png   .png  train  anything   system
            198    688     605       yet/though.png   .png  train      note   number
            199    243     227   describe/road.tiff  .tiff  train       end   number
            <BLANKLINE>
            [200 rows x 7 columns]
            Annotations :
                image_id category_str  category_id  ...   box_y_min   box_width  box_height
            id                                      ...
            0         77        reach           22  ...   45.427512   40.116677  318.073851
            1        137     marriage           15  ...  202.481384  435.389400  475.375279
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {14: 'listen', 15: 'marriage', 22: 'reach'}
            >>> splitted.images.groupby("split")["separate"].value_counts()
            split  separate
            train  likely      27
                   number      27
                   attack      22
                   rest        20
                   entire      17
                   enough      16
                   system      15
                   often       11
                   star         0
                   law          0
            valid  star        27
                   law         18
                   entire       0
                   attack       0
                   rest         0
                   likely       0
                   system       0
                   often        0
                   enough       0
                   number       0
            Name: count, dtype: int64
            >>> splitted.images.groupby("split")["balanced"].value_counts()
            split  balanced
            train  could       21
                   coach       21
                   send        19
                   firm        18
                   end         17
                   anything    14
                   training    14
                   region      11
                   lead        10
                   note        10
            valid  could        8
                   end          7
                   note         6
                   anything     5
                   send         5
                   firm         4
                   training     3
                   region       3
                   lead         2
                   coach        2
            Name: count, dtype: int64
        """
        if len(split_names) <= 1:
            raise ValueError(
                f"Must provide at least 2 split names. Got {split_names} of size"
                f" {len(split_names)} instead."
            )
        if len(target_split_shares) != len(split_names):
            raise ValueError(
                "Size mismatch between 'split_names' and 'split_shares'"
                f" ({len(split_names)} vs {len(target_split_shares)})"
            )
        if sum(target_split_shares) != 1:
            raise ValueError(
                "Split share values must addup to 1. Got"
                f" {sum(target_split_shares)} instead"
            )
        if (
            (not keep_separate_groups)
            or keep_separate_groups == "image_id"
            or keep_separate_groups == ("image_id",)
            or keep_separate_groups == ["image_id"]
        ) and (not keep_balanced_groups):
            print("Using simple random split")
            return self.simple_split(
                input_seed, split_names, target_split_shares, inplace
            )

        keep_separate_groups = groups_to_list(keep_separate_groups)
        keep_balanced_groups = groups_to_list(keep_balanced_groups)

        if keep_balanced_groups_weights is None:
            keep_balanced_groups_weights = [1] * len(keep_balanced_groups)
        else:
            keep_balanced_groups_weights = [*keep_balanced_groups_weights]

        keep_balanced_group_names = get_group_names(keep_balanced_groups)

        keep_balanced_image_groups_indices = [
            i
            for i, name in enumerate(keep_balanced_group_names)
            if (name in self.images.columns and name not in self.annotations.columns)
        ]
        keep_balanced_image_groups = [
            keep_balanced_groups[i] for i in keep_balanced_image_groups_indices
        ]
        keep_balanced_image_groups_weights = [
            keep_balanced_groups_weights[i] for i in keep_balanced_image_groups_indices
        ]

        keep_separate_group_names = get_group_names(keep_separate_groups)
        keep_separate_image_groups = [
            name
            for name in keep_separate_group_names
            if (name in self.images.columns and name not in self.annotations.columns)
        ]

        print("Splitting annotations ...")
        splitted_annotations, splitted_images = split_dataframe(
            root_data=self.images,
            input_data=self.annotations,
            input_seed=input_seed,
            split_names=split_names,
            target_split_shares=target_split_shares,
            keep_separate_groups=keep_separate_groups,
            keep_balanced_groups=keep_balanced_groups,
            keep_balanced_groups_weights=keep_balanced_groups_weights,
            inplace=inplace,
            split_at_root_level=True,
            hist_cost_weight=hist_cost_weight,
            share_cost_weight=share_cost_weight,
            earth_mover_regularization=earth_mover_regularization,
        )

        print("Splitting images ...")
        splitted_images = split_dataframe(
            input_data=splitted_images,
            root_data=None,
            input_seed=input_seed,
            split_names=split_names,
            target_split_shares=target_split_shares,
            keep_separate_groups=keep_separate_image_groups,
            keep_balanced_groups=keep_balanced_image_groups,
            keep_balanced_groups_weights=keep_balanced_image_groups_weights,
            inplace=inplace,
            split_at_root_level=False,
            hist_cost_weight=hist_cost_weight,
            share_cost_weight=share_cost_weight,
            earth_mover_regularization=earth_mover_regularization,
        )

        if inplace:
            self.images = splitted_images
            self.annotations = splitted_annotations
            return self

        return self.from_template(
            images=splitted_images, annotations=splitted_annotations
        )

    def to_parquet(self, output_dir: Path | str, overwrite: bool = False) -> None:
        """Save dataset object to a folder containing parquet files for dataframes
        and a metadata.yaml file for other attributes.

        Note:
            The dataframe dtypes must be serializable as parquet. This includes int,
            float, strings, lists; but not custom objects like e.g.
            :class:`pathlib.Path`

        Args:
            output_dir: folder path where to save the object's attributes.
                If ``overwrite`` is set to False, it must not already exist.
            overwrite: If set to True, will remove the ``output_dir`` directory if it
                already exists. Defaults to False

        See Also:
            :mod:`lours.utils.parquet_saver`
        """
        dict_to_parquet(
            {k: v for k, v in vars(self).items() if not k.startswith("_")}
            | {"__name__": self.__class__.__name__},
            Path(output_dir),
            overwrite=overwrite,
        )

    def to_darknet(
        self,
        output_path: Path | str,
        copy_images: bool = False,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        create_split_folder: bool = False,
    ) -> None:
        """Save dataset in darknet format, readable by
        `darknet <https://github.com/AlexeyAB/darknet>`__ .
        Save in the same folder the images, annotations and data files

        Args:
            output_path: folder where images and annotations will be stored
            copy_images: If set to False,
                will create a symbolic link instead of copying. Much faster,
                but needs to keep original images in the same relative path.
                Defaults to False.
            overwrite_images: if set to False, will skip images that are already copied.
                Defaults to True.
            overwrite_labels: if set to False, will skip annotation that are already
                created. Defaults to True.
            create_split_folder: if set to True, will create a dedicated folder for each
                split and will save images in it. Image paths in {split}.txt will be
                changed accordingly. Note that this changes the dataset structure.
                Defaults to False

        See Also:
            - :mod:`lours.dataset.io.darknet`
            - :meth:`to_yolov5`
            - :meth:`to_yolov7`
        """
        from .io.darknet import dataset_to_darknet

        return dataset_to_darknet(
            self,
            output_path,
            copy_images,
            overwrite_images,
            overwrite_labels,
            yolo_version=1,
            create_split_folder=create_split_folder,
        )

    def to_yolov5(
        self,
        output_path: Path | str,
        copy_images: bool = False,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        split_name_mapping: dict[str, str] | None = None,
        create_split_folder: bool = False,
    ) -> None:
        """Save dataset in format readable by
        `Yolov5 <https://github.com/ultralytics/yolov5>`__ .
        Save each split in its dedicated split file containing paths to corresponding
        images, separate images and annotations with the folders ``images`` and
        ``labels``, and save corresponding info in data.yaml, at the root of the output
        path.

        Optionally, remap the split values so that it fits the training script.
        Normally, yolov5 expect ``train``, ``val`` and ``test`` sets. The default
        mapping replaces ``valid`` and ``validation`` to ``val``, and ``eval`` to
        ``test``.

        Args:
            output_path: folder where images and annotations will be stored
            copy_images: If set to False,
                will create a symbolic link instead of copying. Much faster,
                but needs to keep original images in the same relative path.
                Defaults to False.
            overwrite_images: if set to False, will skip images that are already copied.
                Defaults to True.
            overwrite_labels: if set to False, will skip annotation that are already
                created. Defaults to True.
            split_name_mapping: mapping dict to replace split names to other ones. split
                names not present in mapping will not be modified. If set to None,
                will apply yolov5 conventional mapping, i.e.
                ``{'valid': 'val', 'validation': 'val', 'eval': 'test'}``.
                Defaults to None
            create_split_folder: if set to True, will create a dedicated folder for each
                split and will save images in it. Image paths in {split}.txt will be
                changed accordingly. Note that this changes the dataset structure.
                Defaults to False

        See Also:
            - :mod:`lours.dataset.io.darknet`
            - :meth:`to_darknet`
            - :meth:`to_yolov7`
        """
        from .io.darknet import dataset_to_darknet

        return dataset_to_darknet(
            self,
            output_path,
            copy_images,
            overwrite_images,
            overwrite_labels,
            yolo_version=5,
            split_name_mapping=split_name_mapping,
            create_split_folder=create_split_folder,
        )

    def to_yolov7(
        self,
        output_path: Path | str,
        copy_images: bool = False,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        split_name_mapping: dict[str, str] | None = None,
        create_split_folder: bool = False,
    ) -> None:
        """Save dataset in format readable by
        `Yolov7 <https://github.com/WongKinYiu/yolov7>`__ .
        Save each split in its dedicated split file containing paths to corresponding
        images, separate images and annotations with the folders ``images`` and
        ``labels``, and save corresponding info in data.yaml, at the root of the output
        path.

        Optionally, remap the split values so that it fits the training script.
        Normally, yolov5 expect ``train``, ``val`` and ``test`` sets. The default
        mapping replaces ``valid`` and ``validation`` to ``val``, and ``eval`` to
        ``test``.

        Note:
            The only difference with :func:`.to_yolov5` is the fact that path to split
            list files are absolute and not relative to the yaml file parent folder.

        Args:
            output_path: folder where images and annotations will be stored
            copy_images: If set to False,
                will create a symbolic link instead of copying. Much faster,
                but needs to keep original images in the same relative path.
                Defaults to False.
            overwrite_images: if set to False, will skip images that are already copied.
                Defaults to True.
            overwrite_labels: if set to False, will skip annotation that are already
                created. Defaults to True.
            split_name_mapping: mapping dict to replace split names to other ones. split
                names not present in mapping will not be modified. If set to None,
                will apply yolov5 conventional mapping, i.e.
                ``{'valid': 'val', 'validation': 'val', 'eval': 'test'}``.
                Defaults to None
            create_split_folder: if set to True, will create a dedicated folder for each
                split and will save images in it. Image paths in {split}.txt will be
                changed accordingly. Note that this changes the dataset structure.
                Defaults to False

        See Also:
            - :mod:`lours.dataset.io.darknet`
            - :meth:`to_darknet`
            - :meth:`to_yolov5`
        """
        from .io.darknet import dataset_to_darknet

        return dataset_to_darknet(
            self,
            output_path,
            copy_images,
            overwrite_images,
            overwrite_labels,
            yolo_version=7,
            split_name_mapping=split_name_mapping,
            create_split_folder=create_split_folder,
        )

    def to_coco(
        self,
        output_path: Path | str,
        copy_images: bool = False,
        to_jpg: bool = True,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        add_split_suffix: bool | None = None,
        box_format: str = "XYWH",
    ) -> None:
        """Save dataset in coco format. Will create in output directory one
        JSON file per split present in the dataset.

        Args:
            output_path: Output folder where to save the JSON files
            copy_images: If set to False,
                will create a symbolic link instead of copying. Much faster,
                but needs to keep original images in the same relative path.
                Defaults to False.
            to_jpg: if True, along with previous option, will convert images to jpg if
                needed. Defaults to True.
            overwrite_images: if set to False, will skip images that are already copied.
                Defaults to True.
            overwrite_labels: if set to False, will skip JSON files that are already
                created. Defaults to True.
            add_split_suffix: if set to True, will append the name of the split to the
                json output files. Cannot be False if dataset has multiple splits.
                If not set, will add suffix only if dataset has multiple splits.
            box_format: what type of annotation the json file will have.
                It will be converted from XYWH. Defaults to XYWH

        See Also:
            - :mod:`lours.dataset.io.coco`
        """
        from .io.coco import dataset_to_coco

        return dataset_to_coco(
            self,
            output_path,
            copy_images,
            to_jpg,
            overwrite_images,
            overwrite_labels,
            add_split_suffix,
            box_format=box_format,
        )

    def to_caipy(
        self,
        output_path: Path | str,
        use_schema: bool = False,
        json_schema: Path | str | None = "default",
        copy_images: bool = True,
        to_jpg: bool = True,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        flatten_paths: bool = True,
    ) -> None:
        """Convert dataset to cAIpy format.

        Note:
            - Unless specified otherwise, relative paths of images a flattened during
              the export, which modifies the dataset if the images and annotations
              were stored in subfolders, but will put all images and annotations of a
              particular split in their respective root folder.
            - If schema is not given, the nested dictionary will be deduced from column
              names with the separator "."

        Args:
            output_path: folder where cAIpy folder will be recreated
            use_schema: If set to True, and ``json_schema`` is not None, will use schema
                for validation and formatting (see option ``json_schema``)
            json_schema: Path to a schema that output json dicts will be tested against
                for compliance. They will also be used to remove columns for fields no
                included in the schema. Can be either a url or a path object.
                If set to None, or ``use_schema`` is set to False,
                will not perform any test. Defaults to default schema.
            copy_images: If set to False,
                will create a symbolic link instead of copying. Much faster,
                but needs to keep original images in the same relative path.
                Defaults to False.
            to_jpg: if True, along with previous option, will convert
                images to jpg if needed. Defaults to True.
            overwrite_images: if set to False,
                will skip images that are already copied. Defaults to True.
            overwrite_labels: if set to False,
                will skip annotation that are already created. Defaults to True.
            flatten_paths: if set to True, will put all files in the root Annotations
                and Images folders by replacing folder separation ("/") with "_" in
                relative path. Defaults to False

        See Also:
            - :mod:`lours.dataset.io.caipy`
            - :meth:`to_caipy_generic`
        """  # noqa: E501
        from .io.caipy import dataset_to_caipy

        return dataset_to_caipy(
            self,
            output_path,
            use_schema,
            json_schema,
            copy_images,
            to_jpg,
            overwrite_images,
            overwrite_labels,
            flatten_paths,
        )

    def to_caipy_generic(
        self,
        output_images_folder: Path | str | None,
        output_annotations_folder: Path | str,
        use_schema: bool = False,
        json_schema: Path | str | None = "default",
        copy_images: bool = True,
        to_jpg: bool = True,
        overwrite_images: bool = True,
        overwrite_labels: bool = True,
        flatten_paths: bool = True,
    ) -> None:
        """Convert dataset to cAIpy format, but with the possibility to specify images
        and annotations folders rather than a root folder with Images and Annotations
        sub-folders. It is especially useful when creating predictions or saving
        variations of a annotations.

        Note:
            - Unless specified otherwise, relative paths of images a flattened during
              the export, which modifies the dataset if the images and annotations
              were stored in subfolders, but will put all images and annotations of a
              particular split in their respective root folder.
            - If schema is not given, the nested dictionary will be deduced from column
              names with the separator "."

        Args:
            output_images_folder: root folder where the images will be saved. If None,
                will not save images. Useful when only saving predictions or a
                variations of annotations.
            output_annotations_folder: root folder where the json file will be saved.
            use_schema: If set to True, and ``json_schema`` is not None, will use schema
                for validation and formatting (see option ``json_schema``)
            json_schema: Path to a schema that output json dicts will be tested against
                for compliance. They will also be used to remove columns for fields no
                included in the schema. Can be either a url or a path object.
                If set to None, or ``use_schema`` is set to False,
                will not perform any test. Defaults to default schema.
            copy_images: If set to False, will create a symbolic link instead of
                copying. Much faster, but needs to keep original images in the same
                relative path. Defaults to False.
            to_jpg: if True, will convert images to jpg if needed. Defaults to True.
            overwrite_images: if set to False, will skip images that are already copied.
                Defaults to True.
            overwrite_labels: if set to False, will skip annotation that are already
                created. Defaults to True.
            flatten_paths: if set to True, will put all files in the root Annotations
                and Images folders by replacing folder separation ("/") with "_" in
                relative path. Defaults to True

        See Also:
            - :mod:`lours.dataset.io.caipy`
            - :meth:`to_caipy`

        """  # noqa: E501
        from .io.caipy import dataset_to_caipy_generic

        return dataset_to_caipy_generic(
            self,
            output_images_folder,
            output_annotations_folder,
            use_schema,
            json_schema,
            copy_images,
            to_jpg,
            overwrite_images,
            overwrite_labels,
            flatten_paths,
        )

    def to_fiftyone(
        self,
        dataset_name: str | None = None,
        annotations_name: str = "groundtruth",
        allow_keypoints: bool = False,
        record_fo_ids: bool = False,
        existing: Literal["update", "erase", "error"] = "error",
    ) -> "fo.Dataset":
        """Convert the dataset into a
        :class:`fiftyone dataset <fiftyone.core.dataset.Dataset>`, that can then be
        inspected with Fiftyone's webapp. The resulting dataset will have the sample
        field with the name specified in the argument ``annotations_name``.

        Args:
            dataset_name: Name of the fiftyone dataset to add the samples to.
                If the dataset does not exist, it will be created.
                If set to None, will be the folder name of self.dataset_folder.
                Defaults to None.
            annotations_name: Name of the sample field for the annotations. If the
                dataset already exists, the sample field will be created if it does
                not exist, and it will be merged if it already exists.
            allow_keypoints: if set to True, will convert bounding boxes of size 0 to be
                keypoints instead of detection objects.
            record_fo_ids: whether to record the fiftyone ids of samples and
                annotations. If set to True, will create ``fo_id`` column in self.images
                and ``fo_id`` and ``is_keypoint`` column in self.annotations to be able
                to reindex them in the created fiftyone dataset
            existing: What to do in case there is already a fiftyone dataset with the
                same name.

                - "error": will raise an error.
                - "erase": will erase the existing dataset before uploading
                  this one
                - "update": will try to update the dataset by fusing together samples
                  with the same "relative_path"

                Defaults to "error".

        Returns:
            :class:`fiftyone.core.dataset.Dataset`: Fiftyone one dataset that can then
            be used to launch the webapp with
            ``fiftyone.launch_app(evaluator.to_fiftyone("dataset"))``

        See Also:
            - :ref:`Related tutorial </notebooks/5_demo_fiftyone.ipynb>`
            - :mod:`lours.utils.fiftyone_convert`
        """
        from ..utils.fiftyone_convert import create_fo_dataset

        if (
            self.booleanized_columns["images"]
            or self.booleanized_columns["annotations"]
        ):
            return self.debooleanize().to_fiftyone(
                dataset_name, annotations_name, allow_keypoints, record_fo_ids, existing
            )

        if dataset_name is None:
            if self.dataset_name is None:
                dataset_name = self.images_root.name
            else:
                dataset_name = self.dataset_name
        fo_dataset, fo_image_ids, fo_annotations_ids = create_fo_dataset(
            name=dataset_name,
            images_root=self.images_root,
            images=self.images,
            annotations={annotations_name: self.annotations},
            label_map=self.label_map,
            image_tag_columns=self.get_image_attributes(),
            annotations_attributes_columns=self.get_annotations_attributes(),
            allow_keypoints=allow_keypoints,
            existing=existing,
        )
        if record_fo_ids:
            self.images["fo_id"] = fo_image_ids
            # a simple concat with axis=1 could work, but this snippet is clearer
            # in terms of what column are expected and acts as a check
            self.annotations["fo_id"] = fo_annotations_ids[annotations_name]["fo_id"]
            self.annotations["is_keypoint"] = fo_annotations_ids[annotations_name][
                "is_keypoint"
            ]

        return fo_dataset

    def add_detection_annotation(
        self,
        image_id: int | Sequence[int] | ndarray,
        bbox_coordinates: Sequence[float] | Sequence[Sequence[float]] | ndarray,
        category_id: ndarray | int | Sequence[int],
        format_string: str = "XYWH",
        inplace: bool = False,
        label_map: dict[int, str] | None = None,
        category_ids_mapping: dict[int, int] | None = None,
        confidence: float | ndarray | Sequence[float] | None = None,
        **other_columns: float | str | ndarray | Sequence[float] | Sequence[str],
    ) -> Self:
        """Add one or multiple detection annotations to the current dataset.
        In the case of a single annotation, every option can be a single value, but in
        the case of multiple annotations, every option needs to be an array of such
        values, and every array needs to be the same length.

        Note:
            In additions to the following options, you can add other fields as well,
            with keyword arguments.


        Args:
            image_id: image identifier to link each detection to the corresponding image
            bbox_coordinates: list of coordinates for the bounding box. Can follow any
                compatible format, as long as it is given in the next format
            category_id: category of each detection. Label will be deduced from
                dataset's label map
            format_string: format of coordinates, whether coordinates are relatives,
                using corner points of the box, box dimensions, etc.
                See :func:`.import_bbox` for more info
            inplace: if set to True, will modify the dataset inplace and return self.
                Else, will return a modified Dataset. Defaults to False.
            label_map: In the case the current dataset's label map is incomplete, merge
                it with this new label map. current label map and new label map must be
                compatible, see :func:`.merge_label_maps`. Defaults to None.
            category_ids_mapping: Optional dictionary to map annotated category ids into
                the right ids. This is useful for example when a neural network can only
                use a contiguous label map. Defaults to None
            confidence: Optional field for confidence, in the case annotations are
                actually predictions. Defaults to None.
            **other_columns: kwargs of additional optional fields

        Raises:
            ValueError: raised when giving numpy arrays are not the same number of
                elements, or if the bounding box coordinates is not of the shape either
                [4], or [N, 4]

        Returns:
            A new Dataset with appended annotations to ``self.annotations`` if ``inplace``
            is False, or itself otherwise.

        See Also:
            :mod:`lours.utils.annotations_appender`
            :meth:`annotation_append`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset()
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...  73.932999   71.552480   42.673983
            1          0          why           19  ...   4.567638  248.551257  122.602211
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> example.add_detection_annotation(
            ...     image_id=0,
            ...     bbox_coordinates=[0, 0, 0.5, 0.5],
            ...     format_string="xyxy",
            ...     category_id=14,
            ...     confidence=0.5,
            ... )
            Dataset object containing 2 images and 3 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...   box_width  box_height  confidence
            id                                      ...
            0          0         step           15  ...   71.552480   42.673983         NaN
            1          0          why           19  ...  248.551257  122.602211         NaN
            2          0           14           14  ...  171.000000   68.000000         0.5
            <BLANKLINE>
            [3 rows x 9 columns]
            Label map :
            {14: '14', 15: 'step', 19: 'why', 25: 'interview'}

        """
        from ..utils.annotations_appender import add_detection_annotation

        return add_detection_annotation(
            input_dataset=self,
            image_id=image_id,
            bbox_coordinates=bbox_coordinates,
            format_string=format_string,
            category_id=category_id,
            inplace=inplace,
            label_map=label_map,
            category_ids_mapping=category_ids_mapping,
            confidence=confidence,
            **other_columns,
        )

    def annotation_append(
        self,
        format_string: str = "XYWH",
        category_ids_mapping: dict[int, int] | None = None,
        label_map: dict[int, str] | None = None,
    ) -> "AnnotationAppender":
        """Create a context manager to add detection tensors to the current dataset with
        the :meth:`.AnnotationAppender.append` method, as if the Dataset was a list.
        After the appending is finished, the appender construct big numpy arrays to
        concatenate to the dataset's annotations dataframe.

        Note:
            The dataset object from which this context manager is created is
            modified inplace, similar to a list append.

        Args:
            format_string: format string for incoming bounding boxes. Depend on your
                detector conventions. Defaults to "XYWH".
            category_ids_mapping: Optional dictionary to map annotated category ids into
                the right ids. This is useful for example when a neural network can only
                use a contiguous label map. Defaults to None
            label_map: Optional label map for objects outside the current label map.
                Must be compatible with the current label map (i.e. no category id
                clash). Defaults to None.

        Returns:
            :class:`.AnnotationAppender`: Context
            manager to easily add detection tensors

        See Also:
            :mod:`lours.utils.annotations_appender`
            :meth:`add_detection_annotation`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> example = dummy_dataset()
            >>> example
            Dataset object containing 2 images and 2 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_y_min   box_width  box_height
            id                                      ...
            0          0         step           15  ...  73.932999   71.552480   42.673983
            1          0          why           19  ...   4.567638  248.551257  122.602211
            <BLANKLINE>
            [2 rows x 8 columns]
            Label map :
            {15: 'step', 19: 'why', 25: 'interview'}
            >>> with example.annotation_append(
            ...     format_string="xyxy", label_map={1: "new_class"}
            ... ) as appender:
            ...     appender.append(
            ...         image_id=0,
            ...         bbox_coordinates=np.array([[0, 0, 0.5, 0.5]]),
            ...         category_id=15,
            ...         confidence=0.5,
            ...         other_attribute=0,
            ...     )
            ...     appender.append(
            ...         image_id=[1, 0],
            ...         bbox_coordinates=np.array(
            ...             [[0.1, 0.1, 0.9, 0.9], [0.2, 0.3, 0.5, 0.5]]
            ...         ),
            ...         category_id=np.array([1, 15]),
            ...         confidence=np.array([0.2, 0.3]),
            ...         other_attribute=np.array([3, 4]),
            ...     )
            >>> example
            Dataset object containing 2 images and 5 objects
            Name :
                inside_else_memory
            Images root :
                such/serious
            Images :
                width  height      relative_path   type  split
            id
            0     342     136       help/me.jpeg  .jpeg  train
            1     377     167  whatever/wait.png   .png  train
            Annotations :
                image_id category_str  category_id  ...  box_height  confidence  other_attribute
            id                                      ...
            0          0         step           15  ...   42.673983         NaN              NaN
            1          0          why           19  ...  122.602211         NaN              NaN
            2          0         step           15  ...   68.000000         0.5              0.0
            3          1    new_class            1  ...  133.600000         0.2              3.0
            4          0         step           15  ...   27.200000         0.3              4.0
            <BLANKLINE>
            [5 rows x 10 columns]
            Label map :
            {1: 'new_class', 15: 'step', 19: 'why', 25: 'interview'}
        """
        from ..utils.annotations_appender import AnnotationAppender

        return AnnotationAppender(
            self,
            format_string=format_string,
            category_ids_mapping=category_ids_mapping,
            label_map=label_map,
        )
