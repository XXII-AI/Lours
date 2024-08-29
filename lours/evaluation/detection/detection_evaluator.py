from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lours.dataset import Dataset

from ...utils.grouper import get_group_names, group_list, group_relational_data
from .detection_evaluator_base import DetectionEvaluatorBase
from .util import (
    compute_average_precision,
    confusion_matrix,
    construct_matches_df,
    pr_curve,
)

if TYPE_CHECKING:
    pass


class DetectionEvaluator(DetectionEvaluatorBase):
    """Class specialization for detection tasks Note that the constructor is the
    same as the base Evaluator

    See Also:
        :ref:`related tutorial </notebooks/3_demo_evaluation_detection.ipynb>`
    """

    matches: dict[str, dict[str, pd.DataFrame]] = {
        "category_specific": {},
        "category_agnostic": {},
    }
    """Nested dictionary of DataFrames containing matched bounding boxes between
    groundtruth and corresponding prediction, depending on the way of computing
    matches (between all categories or between similar categories). Note that the
    the sub dictionaries will be empty until :func:`compute_matches` is called"""

    def __init__(
        self, groundtruth: Dataset, name: str | None = None, **predictions: Dataset
    ):
        """Constructor of the DetectionEvaluator class.
        The only difference with vanilla :class:`.Evaluator` is the matches that is set
        to an empty dictionary.
        """
        self.matches = {
            "category_specific": {},
            "category_agnostic": {},
        }
        super().__init__(groundtruth, name, **predictions)

    def compute_matches(
        self,
        predictions_names: str | Iterable[str] | None = None,
        min_iou: float = 0,
        category_agnostic: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Get matches between predictions and targets of the Evaluator.

        See Also:
            :ref:`Related tutorial </notebooks/3_demo_evaluation_detection.ipynb#Compute-the-matches>`

        Args:
            predictions_names: name or collection of prediction names to compute the
                matches on. If set to None, will compute the matches with the prediction
                DataFrames contained in the ``self.predictions_dictionary`` attribute.
                Defaults to None
            min_iou: IoU above which the detection is considered
                valid. Defaults to 0. Note that the lower bound of min_iou is not
                inclusive.
            category_agnostic: if set to False, matches are computed between categories,
                otherwise, matches are computed globally

        Returns:
            dict of DataFrame of matches, one entry per prediction specified in
            arguments. Will contain ``prediction_id`` and ``groundtruth_id`` columns.
            Index is unrelevant. Each prediction id and target id should appear once and
            only once. As such, at worse (no match at all), the dataframe will
            be :math:`N+M` rows with :math:`N` the number of predictions and :math:`M`
            the number of targets, and at best it will be :math:`max(M,N)`

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> groundtruth = dummy_dataset(
            ...     10,
            ...     1000,
            ...     label_map={0: "person", 1: "car"},
            ... )
            >>> predictions1 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=0,
            ... )
            >>> predictions2 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=1,
            ... )
            >>> evaluator = DetectionEvaluator(
            ...     groundtruth=groundtruth, A=predictions1, B=predictions2
            ... )
            >>> matches = evaluator.compute_matches()
            computing matches between groundtruth and A (category specific)
            computing matches between groundtruth and B (category specific)
            >>> matches["A"]
                  prediction_id       iou  groundtruth_id
            0             2311  0.370857             207
            1              515  0.586261             820
            2             7071  0.468022             585
            3             4444  0.089832              87
            4              235  0.431787             105
            ..             ...       ...             ...
            487           5016  0.000000            <NA>
            488           3608  0.000000            <NA>
            489            437  0.000000            <NA>
            490           8837  0.000000            <NA>
            491           2508  0.000000            <NA>
            <BLANKLINE>
            [10000 rows x 3 columns]

            You can select a particular set of prediction to only compute them

            >>> B_matches = evaluator.compute_matches(
            ...     predictions_names="B", category_agnostic=True
            ... )
            computing matches between groundtruth and B (category agnostic)
            >>> B_matches["B"]
                   prediction_id       iou  groundtruth_id
            0               7849  0.267152             832
            1               8819  0.089308             130
            2               6537  0.322729             785
            3               1616  0.406822             326
            4               8021  0.510778             929
            ...              ...       ...             ...
            1022            7377  0.000000            <NA>
            1023            8370  0.000000            <NA>
            1024            3534  0.000000            <NA>
            1025            7087  0.000000            <NA>
            1026            1410  0.000000            <NA>
            <BLANKLINE>
            [10000 rows x 3 columns]
        """
        if isinstance(predictions_names, str):
            predictions_names = [predictions_names]
        if predictions_names is None:
            predictions_names = list(self.predictions_dictionary)
        groundtruth = self.groundtruth.assign(groundtruth=True)
        matches = {}
        for prediction_name in predictions_names:
            print(
                f"computing matches between groundtruth and {prediction_name} (category"
                f" {'agnostic' if category_agnostic else 'specific'})"
            )
            prediction = self.predictions_dictionary[prediction_name].assign(
                groundtruth=False
            )
            groundtruth_prediction = pd.concat([groundtruth, prediction])

            groups = ["image_id"]
            tqdm.pandas()
            if category_agnostic:
                tqdm.pandas()
                matches = self.matches["category_agnostic"]
            else:
                matches = self.matches["category_specific"]
                groups.append("category_id")
            grouped = groundtruth_prediction.groupby(groups, group_keys=False)
            matches[prediction_name] = grouped.progress_apply(  # pyright: ignore
                partial(construct_matches_df, min_iou=min_iou), include_groups=False
            )
        return matches

    def compute_confusion_matrix(
        self,
        predictions_names: str | Iterable[str] | None = None,
        groups: group_list = (),
        min_iou: float = 0,
        min_confidence: float = 0,
    ) -> pd.DataFrame:
        """Compute confusion matrix to evaluate object detection.

        See Also:
            :ref:`related tutorial </notebooks/3_demo_evaluation_detection.ipynb#Computing-confusion-matrix>`

        Args:
            predictions_names: name or collection of prediction names to compute the
                matches on. If set to None, will compute the matches with the prediction
                DataFrames contained in the ``self.predictions_dictionary`` attribute.
                Default to None
            groups: Groups of image or annotation attributes to use to
                partition evaluation results to compute multiple confusion matrices.
                Must be a :obj:`.group_list` . Defaults to ()
            min_iou: IoU above which the detection is considered
                valid. Defaults to 0. Note that the lower bound of min_iou is not
                inclusive.
            min_confidence: confidence threshold above which the detection is considered
                valid. Defaults to 0. Note that the lower bound of min_confidence is not
                inclusive.

        Returns:
            A Dataframe with confusion data for each group name(s) (if any)
            of each predictions_names.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> groundtruth = dummy_dataset(
            ...     10,
            ...     1000,
            ...     label_map={0: "person", 1: "car"},
            ... )
            >>> predictions1 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=0,
            ... )
            >>> predictions2 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=1,
            ... )
            >>> evaluator = DetectionEvaluator(
            ...     groundtruth=groundtruth, A=predictions1, B=predictions2
            ... )
            >>> evaluator.compute_confusion_matrix().reset_index().set_index(
            ...     ["model", "label"]
            ... )
            computing matches between groundtruth and A (category agnostic)
            computing matches between groundtruth and B (category agnostic)
            Processing confusion matrix for model=A
            Processing confusion matrix for model=B
                               car    person  None
            model label
            A     car     0.487179  0.512821   0.0
                  person  0.470588  0.529412   0.0
                  None    0.500889  0.499111   0.0
            B     car     0.495069  0.504931   0.0
                  person  0.470588  0.529412   0.0
                  None    0.505556  0.494444   0.0

            You can also use a minimum iou and select a subset of all prediction sets.

            >>> evaluator.compute_confusion_matrix(
            ...     min_iou=0.1, predictions_names="A"
            ... ).drop("model", axis=1)
            Processing confusion matrix for model=A
                         car    person      None
            label
            car     0.362919  0.374753  0.262327
            person  0.312373  0.367140  0.320487
            None    0.500377  0.499623  0.000000
        """
        if isinstance(predictions_names, str):
            predictions_names = [predictions_names]
        if predictions_names is None:
            predictions_names = list(self.predictions_dictionary)

        # compute category agnostic matches between prediction and groundtruth, and
        # construct a dataframe with groundtruth and prediction id, along with
        # category name for groundtruth (groundtruth_label) and prediction
        # (prediction_label). Groundtruth ids not matched to any prediction will have
        # a prediction label set to None (false negative) and prediction ids not matches
        # to any groundtruth will have a prediction label set to None (false positive).
        result_dict = {}
        group_names = get_group_names(groups)
        for name in predictions_names:
            if name not in self.matches["category_agnostic"]:
                self.compute_matches(name, category_agnostic=True)

            matches = self.matches["category_agnostic"][name]
            results = self.groundtruth.reset_index(names="groundtruth_id")
            results = pd.merge(
                results,
                matches.dropna(subset=["groundtruth_id"]),
                right_on="groundtruth_id",
                left_on="groundtruth_id",
            )
            results["confidence"] = 0.0
            results = results.rename(columns={"category_str": "groundtruth_label"})
            detected = results["prediction_id"].dropna()
            current_predictions = self.predictions_dictionary[name]
            results.loc[~results["prediction_id"].isna(), "confidence"] = (
                current_predictions.loc[detected, "confidence"].values
            )
            results.loc[~results["prediction_id"].isna(), "prediction_label"] = (
                current_predictions.loc[detected, "category_str"].values
            )

            # Then add the unmatched predictions
            false_positive = matches.loc[
                matches["groundtruth_id"].isna(), "prediction_id"
            ]
            results_fp = (
                current_predictions.loc[false_positive]
                .reset_index()
                .rename(
                    columns={"id": "prediction_id", "category_str": "prediction_label"}
                )
            )
            results_fp = pd.merge(
                results_fp,
                matches[matches["groundtruth_id"].isna()],
                right_on="prediction_id",
                left_on="prediction_id",
            )
            results = pd.concat([results, results_fp], ignore_index=True)

            # Keep matches with IOU above threshold or prediction and groundtruth which
            # have not been matched
            iou_above_threshold = (results["iou"] > min_iou) | (results["iou"] == 0)

            # Matches with IOU below threshold are duplicated to be considered
            # as False Negative and False Positive
            iou_below_threshold_gt = (results["iou"] <= min_iou) & results[
                ["groundtruth_label", "prediction_label"]
            ].notna().all(axis=1)
            iou_below_threshold_pred = iou_below_threshold_gt

            df_iou_above_threshold = results[iou_above_threshold]
            df_iou_below_threshold_gt = results[iou_below_threshold_gt].assign(
                prediction_label=pd.NA
            )
            df_iou_below_threshold_pred = results[iou_below_threshold_pred].assign(
                groundtruth_label=pd.NA
            )

            results = pd.concat(
                [
                    df_iou_above_threshold,
                    df_iou_below_threshold_gt,
                    df_iou_below_threshold_pred,
                ],
                ignore_index=True,
            )

            # Set predictions to None if confidence score lower or equal to
            # min_confidence argument
            results.loc[results["confidence"] <= min_confidence, "prediction_label"] = (
                pd.NA
            )
            # Remove rows where both prediction_label and groundtruth_label are None
            both_none_labels = (
                results[["prediction_label", "groundtruth_label"]].isna()
            ).all(axis=1)
            results = results[~both_none_labels]

            group_dict, category_groups, continuous_groups = group_relational_data(
                results, groups
            )
            pandas_groups = [group_dict[name] for name in group_names]
            result_dict[name] = (results, pandas_groups)

        confusion_dataframes = []
        for p_name, (current_results, pandas_groups) in result_dict.items():
            print(f"Processing confusion matrix for model={p_name}")
            if pandas_groups:
                grouped_data = current_results.groupby(
                    pandas_groups, observed=False
                ).apply(confusion_matrix)
                grouped_data["model"] = p_name
            else:
                grouped_data = confusion_matrix(current_results)
                grouped_data["model"] = p_name
            confusion_dataframes.append(grouped_data)
        return pd.concat(confusion_dataframes, ignore_index=False)

    def compute_precision_recall(
        self,
        predictions_names: str | Iterable[str] | None = None,
        groups: group_list = ("category_id",),
        ious: float | Iterable[float] = (0.0,),
        index_column: str | None = "recall",
        index_values: Iterable[float] | None = None,
        f_scores_betas: Iterable[float] = (1,),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""Compute Precision Recall curves, along with Average precision, with respect
        to recall, for different minimum IoU values.

        The dataset can be grouped, so that you get multiple pr curves in the end.

        It can be either groups of images (applied on ``self.images``) or groups of bbox
        (applied on ``self.groundtruth`` and ``self.predictions_dictionary``).

        In the case the data is not categorical, you must provide the number of
        desired bins of desired bin boundaries, and the cut method will be used to
        construct groups.

        See Also:
            - :ref:`Related tutorial </notebooks/3_demo_evaluation_detection.ipynb#Computing-AP-+-Yolov5-metrics>`
            - :class:`.ContinuousGroup`
            - :func:`pandas.cut` and :func:`pandas.qcut` for continuous groups.

        Note:
            For bbox groups, the value used will be the one of the target, except for
            false positive (no matching target) where the prediction data will be used.
            For example, the bbox size used for grouping will be the target one and not
            the prediction. So even if the prediction is out of bound, the detection
            will be considered valid as long as the IoU is high enough.
            However, when there is a false positive, the size of prediction will be used
            to decide in which group the precision needs to be decreased

        Args:
            predictions_names: names of predictions DataFrames, contained in
                ``self.predictions_dictionary`` to compute the PR curves on. If set to
                None, will compute PR curves for all predictions DataFrames.
            groups: Groups of image or annotation attributes to use to
                partition evaluation results to compute multiple PR curves. Must be a
                :obj:`.group_list` . Defaults to ``("category_id", )``.
            ious: minimum IoU values above which detection are considered valid.
                The higher, the harder it is for a detection to be valid.
                Defaults to 0.
            index_column: If set, will force the values of given column to be in the
                same bins. This will decrease data granularity, but make it possible to
                us this column as index. If not set, each category will have its own
                values, set exactly where recall and precision changes, making the curve
                more precise. Possible arguments are the only monotonous values (either
                increasing or decreasing), i.e. ``recall``, ``precision`` and
                ``confidence_threshold``. Defaults to ``recall`` to match pycocotools
                and fiftyone evaluation workflows.
            index_values: Iterable of bins, increasing float values from 0 to 1. used to
                reindex the dataframe. If set to None, will be 101 points evenly spaced
                from 0 to 1, to match pycocotools and fiftyone evaluation workflows.
                Defaults to None.
            f_scores_betas: beta values to compute the corresponding :math:`F_\beta`
                values in addition to precision and recall.

        Returns:
            PR curve dataset and corresponding average precision.
            The PR curve dataframe will have at least ``precision``,
            ``recall``, ``confidence_threshold``, and ``iou_threshold`` columns, plus
            the :math:`F_\beta` score columns, plus all the columns from the given
            groups. The AP dataframe will have at least AP and iou_threshold columns,
            plus all the columns from the given groups.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> groundtruth = dummy_dataset(
            ...     10,
            ...     1000,
            ...     label_map={0: "person", 1: "car"},
            ...     n_attribute_columns_images={"attribute": 2},
            ... )
            >>> predictions1 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=0,
            ... )
            >>> predictions2 = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     add_confidence=True,
            ...     seed=1,
            ... )
            >>> evaluator = DetectionEvaluator(
            ...     groundtruth=groundtruth, A=predictions1, B=predictions2
            ... )

            Get the Precision Recall curves and the Average Precision dataframe

            >>> pr, ap = evaluator.compute_precision_recall(ious=[0, 0.5])
            computing matches between groundtruth and A (category specific)
            computing matches between groundtruth and B (category specific)
            Processing PR curves for 2 IoU values and 2 prediction sets
            Processing PR curve for model=A and IOU=0
            Processing PR curve for model=B and IOU=0
            Processing PR curve for model=A and IOU=0.5
            Processing PR curve for model=B and IOU=0.5
            >>> ap
               category_id  iou_threshold model        AP category_str
            0            1            0.0     A  0.939509          car
            1            0            0.0     A  0.961933       person
            2            1            0.0     B  0.956845          car
            3            0            0.0     B  0.946684       person
            4            1            0.5     A  0.040764          car
            5            0            0.5     A  0.026722       person
            6            1            0.5     B  0.025094          car
            7            0            0.5     B  0.025750       person
            >>> pr
                 category_id  recall  precision  ...  iou_threshold  model  category_str
            0              1    0.00   1.000000  ...            0.0      A           car
            1              1    0.01   1.000000  ...            0.0      A           car
            2              1    0.02   1.000000  ...            0.0      A           car
            3              1    0.03   0.985714  ...            0.0      A           car
            4              1    0.04   0.985714  ...            0.0      A           car
            ..           ...     ...        ...  ...            ...    ...           ...
            803            0    0.96   0.000000  ...            0.5      B        person
            804            0    0.97   0.000000  ...            0.5      B        person
            805            0    0.98   0.000000  ...            0.5      B        person
            806            0    0.99   0.000000  ...            0.5      B        person
            807            0    1.00   0.000000  ...            0.5      B        person
            <BLANKLINE>
            [808 rows x 8 columns]

            For each class, iou and model, get the confidence threshold with the best f1
            and print the corresponding f1, recall and precision

            >>> best_f1 = pr.groupby(["model", "iou_threshold", "category_id"])[
            ...     "f1_score"
            ... ].idxmax()
            >>> pr.loc[best_f1, ["f1_score", "recall", "precision"]].set_index(
            ...     best_f1.index
            ... )
                                             f1_score  recall  precision
            model iou_threshold category_id
            A     0.0           0            0.904181    0.89   0.920502
                                1            0.884258    0.88   0.888889
                  0.5           0            0.131444    0.10   0.194030
                                1            0.158687    0.13   0.208202
            B     0.0           0            0.883191    0.87   0.903158
                                1            0.898718    0.90   0.898039
                  0.5           0            0.124986    0.10   0.168350
                                1            0.131136    0.10   0.203922

            Use a grouper to have PR values with respect to "attribute" image column,
            box height columns, thanks to a continuous group.

            >>> from lours.utils.grouper import ContinuousGroup
            >>> height_group = ContinuousGroup("box_height", bins=2, qcut=True)
            >>> pr, ap = evaluator.compute_precision_recall(
            ...     ious=0.1,
            ...     groups=["attribute", "category_id", height_group],
            ...     predictions_names="B",
            ... )
            Processing PR curves for 1 IoU value and 1 prediction set
            Processing PR curve for model=B and IOU=0.1
            >>> ap.set_index(["box_height", "attribute", "category_str"])["AP"]
            box_height          attribute  category_str
            (209.059, 938.398]  return     car             0.687178
            (0.0295, 209.059]   return     car             0.486098
                                to         person          0.394769
            (209.059, 938.398]  to         car             0.670351
                                return     person          0.727590
            (0.0295, 209.059]   to         car             0.517899
                                return     person          0.372749
            (209.059, 938.398]  to         person          0.586228
            Name: AP, dtype: float64


        """
        if isinstance(predictions_names, str):
            predictions_names = [predictions_names]
        if predictions_names is None:
            predictions_names = list(self.predictions_dictionary)
        for name in predictions_names:
            if name not in self.matches["category_specific"]:
                self.compute_matches(name, category_agnostic=False)

        if index_column is not None:
            if index_values is None:
                index_values = np.linspace(0, 1, 101)
            else:
                index_values = list(index_values)
            reindex_series = pd.Series(index_values, name=index_column)
        else:
            reindex_series = None

        assert self.matches

        result_dict = {}
        group_names = get_group_names(groups)
        for p_name in predictions_names:
            # Construct a dataframe of all targets, with corresponding matching
            # prediction (if any), and confidence from predictions (0 if false negative)
            matches = self.matches["category_specific"][p_name]
            current_predictions = self.predictions_dictionary[p_name]
            results = self.groundtruth.reset_index(names="groundtruth_id")
            results = pd.merge(
                results,
                matches.dropna(subset=["groundtruth_id"]),
                right_on="groundtruth_id",
                left_on="groundtruth_id",
            )
            results["confidence"] = 0.0
            detected = results["prediction_id"].dropna()
            results.loc[~results["prediction_id"].isna(), "confidence"] = (
                current_predictions.loc[detected, "confidence"].values
            )

            # Then add the unmatched predictions
            false_positive = matches.loc[
                matches["groundtruth_id"].isna(), "prediction_id"
            ]
            results_fp = (
                current_predictions.loc[false_positive]
                .reset_index()
                .rename(columns={"id": "prediction_id"})
            )
            results_fp = pd.merge(
                results_fp,
                matches[matches["groundtruth_id"].isna()],
                right_on="prediction_id",
                left_on="prediction_id",
            )
            results = pd.concat([results, results_fp], ignore_index=True)
            # groundtruth is True if groundtruth_id is NA, False otherwise
            results["groundtruth"] = ~results["groundtruth_id"].isna()

            results = results.sort_values("confidence", ascending=False)

            group_dict, *_ = group_relational_data(results, groups, self.images)
            pandas_groups = [group_dict[name] for name in group_names]
            result_dict[p_name] = (results, pandas_groups)

        precision_recall_curves = []
        if isinstance(ious, float | int):
            ious = [float(ious)]
        else:
            ious = [*ious]
        plural_ious = "s" if len(ious) > 1 else ""
        plural_predictions = "s" if len(result_dict) > 1 else ""
        print(
            f"Processing PR curves for {len(ious)} IoU value{plural_ious} and"
            f" {len(result_dict)} prediction set{plural_predictions}"
        )
        for iou in tqdm(ious):
            for p_name, (results, pandas_groups) in result_dict.items():
                print(f"Processing PR curve for model={p_name} and IOU={iou}")
                if pandas_groups:
                    precision_recall_curve = results.groupby(
                        pandas_groups, sort=False
                    ).progress_apply(  # pyright: ignore
                        partial(
                            pr_curve,
                            min_iou=iou,
                            reindex_series=reindex_series,
                            betas=f_scores_betas,
                        ),
                        include_groups=False,
                    )
                    # Groups are currently in the multiIndex. Reset it to make the
                    # dataframe easier to use: rename the index with the group names and
                    # make it dataframe columns.
                    # Also, the irrelevant id column at the same time
                    precision_recall_curve = precision_recall_curve.rename_axis(
                        [
                            *group_names,
                            "id",
                        ]
                    )
                    precision_recall_curve = precision_recall_curve.reset_index(
                        level=group_names
                    )
                else:
                    precision_recall_curve = pr_curve(
                        results, min_iou=iou, reindex_series=reindex_series
                    )
                precision_recall_curve["iou_threshold"] = iou
                precision_recall_curve["model"] = p_name
                precision_recall_curves.append(precision_recall_curve)

        precision_recall_curves = pd.concat(precision_recall_curves, ignore_index=True)
        average_precisions = precision_recall_curves.groupby(
            group_names + ["iou_threshold", "model"], sort=False
        ).apply(compute_average_precision, include_groups=False)
        average_precisions = average_precisions.rename("AP").reset_index()
        if "category_id" in group_names:
            precision_recall_curves["category_str"] = precision_recall_curves[
                "category_id"
            ].replace(self.label_map)
            average_precisions["category_str"] = average_precisions[
                "category_id"
            ].replace(self.label_map)
        return precision_recall_curves, average_precisions

    def _get_widgets(self):
        import ipywidgets as widgets
        from IPython.display import display

        components_widgets = super()._get_widgets()

        if self.matches:
            for category_name, matches_dict in self.matches.items():
                for prediction_name, matches in matches_dict.items():
                    widget_matches = widgets.Output()

                    # render in output widgets
                    with widget_matches:
                        display(matches)
                    components_widgets[
                        f"{category_name} {prediction_name} Matches (class)"
                    ] = widget_matches

        return components_widgets
