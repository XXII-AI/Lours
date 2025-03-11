"""Module dedicated to Dataset indexers, to be able to index Dataset with pandas style
loc and iloc methods
"""

from typing import Any, Generic, Literal, TypeVar

import pandas as pd

from lours.dataset import Dataset

D = TypeVar("D", bound=Dataset)


class DatasetImLocator(Generic[D]):
    """Locator class dedicated to index a dataset by its images as if we used pandas
    indexing methods on ``dataset.images`` and filtered annotations accordingly.

    Usually used in the context of :meth:`.Dataset.loc` and :meth:`.Dataset.iloc`
    """

    def __init__(self, dataset: D, mode: Literal["loc", "iloc"] = "loc") -> None:
        """Constructor

        Args:
            dataset: Dataset object to index
            mode: whether to use ``dataset.images.loc`` or ``dataset.images.iloc``.
                Defaults to "loc"
        """
        self.dataset = dataset
        self.mode = mode

    def __getitem__(self, index: Any) -> D:
        """Index the dataset with an index applied on ``dataset.images``

        Args:
            index: index object expected to be compatible with ``dataset.images.loc`` or
                ``dataset.images.iloc``.

        Returns:
            Indexed sub-dataset
        """
        if self.mode == "loc":
            new_images = self.dataset.images.loc[index]
        else:
            new_images = self.dataset.images.iloc[index]
        if isinstance(new_images, pd.Series):
            # The Series object indicates only one row. Convert it back to a frame
            new_images = new_images.to_frame().T.astype(self.dataset.images.dtypes)
        new_annotations = self.dataset.annotations.loc[
            self.dataset.annotations["image_id"].isin(new_images.index)
        ]
        return self.dataset.from_template(
            images=new_images,
            annotations=new_annotations,
            reset_booleanized=False,
        )


class DatasetAnnotLocator(Generic[D]):
    """Locator class dedicated to index a dataset by its annotations as if we used
    pandas indexing methods on ``dataset.annotations`` and potentially filtered
    empty images accordingly.

    Usually used in the context of :meth:`.Dataset.loc_annot` and
    :meth:`.Dataset.iloc_annot`
    """

    def __init__(
        self,
        dataset: D,
        mode: Literal["loc", "iloc"] = "loc",
        remove_emptied_images: bool = False,
    ) -> None:
        """Constructor

        Args:
           dataset: Dataset object to index
           mode: whether to use ``dataset.annotations.loc`` or
               ``dataset.annotations.iloc``. Defaults to "loc"
           remove_emptied_images: If se to True, will remove images that no longer have
               annotations, but will keep images that were already empty before.
               Defaults to False
        """
        self.dataset = dataset
        self.mode = mode
        self.remove_emptied_images = remove_emptied_images

    def __getitem__(self, index: Any) -> D:
        """Index the dataset with an index applied on ``dataset.annotations``

        Args:
            index: index object expected to be compatible with
                ``dataset.annotations.loc`` or ``dataset.annotations.iloc``.

        Returns:
            Indexed sub-dataset
        """
        if self.mode == "loc":
            new_annotations = self.dataset.annotations.loc[index]
        else:
            new_annotations = self.dataset.annotations.iloc[index]
        if isinstance(new_annotations, pd.Series):
            new_annotations = new_annotations.to_frame().T.astype(
                self.dataset.annotations.dtypes
            )
        if self.remove_emptied_images:
            already_empty_images = ~self.dataset.images.index.isin(
                self.dataset.annotations["image_id"]
            )
            already_empty_images_ids = self.dataset.images.index[
                already_empty_images
            ].tolist()
            remaining_images = new_annotations["image_id"].unique().tolist()
            new_images = self.dataset.images.loc[
                [
                    *already_empty_images_ids,
                    *remaining_images,
                ]
            ]
        else:
            new_images = self.dataset.images
        return self.dataset.from_template(
            images=new_images,
            annotations=new_annotations,
            reset_booleanized=False,
        )
