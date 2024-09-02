from typing import TYPE_CHECKING, Literal

from ..evaluator import Evaluator

if TYPE_CHECKING:
    import fiftyone as fo


class DetectionEvaluatorBase(Evaluator):
    """Base function for detection based evaluation. can be derived into basic
    detection or crowd detection
    """

    def to_fiftyone(
        self,
        name: str | None = None,
        record_fo_ids: bool = False,
        existing: Literal["error", "update", "erase"] = "error",
    ) -> "fo.Dataset":
        """Convert evaluator to fiftyone.

        Convert the detection evaluator into a
        :class:`fiftyone dataset <fiftyone.core.dataset.Dataset>`, that can then be
        inspected with Fiftyone's webapp. The resulting dataset will have the
        groundtruth sample field, along with all the prediction set's name and value in
        the ``self.predictions_dictionary`` attribute

        Args:
            name: Name of the fiftyone dataset to add the samples to. If the dataset
                does not exist, it will be created. If set to None, will use
                self.image_root folder name
            record_fo_ids: whether to record the fiftyone ids of samples and
                annotations. If set to True, will create ``fo_id`` column in self.images
                and ``fo_id`` and ``is_keypoint`` column in dataframes contained in
                self.predictions_dictionary and self.groundtruth to be able
                to reindex them in the created fiftyone dataset.
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
        from ...utils.fiftyone_convert import create_fo_dataset

        if name is None:
            if self.name is None:
                name = self.images_root.name
            else:
                name = self.name

        fo_dataset, fo_image_ids, fo_annotations_ids = create_fo_dataset(
            name=name,
            images_root=self.images_root,
            images=self.images,
            annotations={
                "groundtruth": self.groundtruth,
                **self.predictions_dictionary,
            },
            label_map=self.label_map,
            image_tag_columns=self.get_image_attributes(),
            annotations_attributes_columns={
                "groundtruth": self.get_annotations_attributes(),
                **{
                    name: self.get_annotations_attributes(name)
                    for name in self.predictions_dictionary
                },
            },
            existing=existing,
        )

        if record_fo_ids:
            self.images["fo_id"] = fo_image_ids
            for name, predictions_frame in self.predictions_dictionary.items():
                # a simple concat with axis=1 could work, but this snippet is clearer
                # in terms of what column are expected and acts as a check
                predictions_frame["fo_id"] = fo_annotations_ids[name]["fo_id"]
                predictions_frame["is_keypoint"] = fo_annotations_ids[name][
                    "is_keypoint"
                ]
            self.groundtruth["fo_id"] = fo_annotations_ids["groundtruth"]["fo_id"]
            self.groundtruth["is_keypoint"] = fo_annotations_ids["groundtruth"][
                "is_keypoint"
            ]

        return fo_dataset
