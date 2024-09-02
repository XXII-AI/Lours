from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...utils.grouper import (
    get_group_names,
    group_list,
    group_relational_data,
    groups_to_list,
)
from .detection_evaluator_base import DetectionEvaluatorBase
from .util import resample_count

if TYPE_CHECKING:
    pass


class CrowdDetectionEvaluator(DetectionEvaluatorBase):
    """Class specialization for crowd detection and counting tasks.
    Note that the constructor is the same as the base Evaluator

    See Also:
        :ref:`related tutorial </notebooks/4_demo_evaluation_crowd.ipynb>`
    """

    def compute_count_error(
        self,
        groups: group_list = "category_id",
        quantiles: Iterable[float] = (0.25, 0.5, 0.75),
        confidence_index: Iterable[float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Count error metrics, both absolute (in number of objects found) and
        relative (with respect to groundtruth number of objects) with respect to
        confidence threshold.

        Along with these metrics, it computes standard deviation of absolute/relative
        error and quantiles for error values.

        See Also:
            :ref:`related tutorial </notebooks/4_demo_evaluation_crowd.ipynb>`

        Computed metrics:
            - Mean Absolute Error (MAE)
            - Root of Mean Square Error (RMSE)
            - Mean Relative Error (MRE)
            - Root of Mean Square Relative Error (RMSRE)

        Args:
            groups: Groups of image or annotation attributes to use to
                partition evaluation results to compute multiple PR curves. Must be a
                :obj:`.group_list` . Defaults to "category_id".
            quantiles: quantile values to get with respect to confidence threshold,
                aggregated per image. Must contain the median value (i.e. 0.5).
                Defaults to (0.25, 0.5, 0.75).
            confidence_index: sequence of confidence thresholds to compute the metric
                on. If set to None, will be 101 equidistant points, from 0 to 1.
                Defaults to None.

        Returns:
            A pair of DataFrames.

            - a DataFrame with computed metrics, with multiindex columns, for
              absolute and relative metrics with respect to confidence
            - a DataFrame with detailed error values with respect to confidence for each
              image, in order to compute statistics manually.

        Example:
            >>> from lours.utils.doc_utils import dummy_dataset
            >>> groundtruth = dummy_dataset(
            ...     10, 1000, label_map={0: "person", 1: "car"}, keypoints_share=1
            ... )
            >>> predictions = dummy_dataset(
            ...     10,
            ...     10000,
            ...     label_map=groundtruth.label_map,
            ...     images=groundtruth.images,
            ...     keypoints_share=1,
            ...     add_confidence=True,
            ... )
            >>> evaluator = CrowdDetectionEvaluator(
            ...     groundtruth=groundtruth, predictions=predictions
            ... )
            >>> errors, detailed = evaluator.compute_count_error()
            >>> errors
                                   absolute              ...  relative
                                        MAE        RMSE  ...     q0.75        model
            category_id confidence                       ...
            0           0.00          452.0  452.378824  ...  9.913239  predictions
                        0.01          447.8  448.170057  ...  9.779314  predictions
                        0.02          442.3  442.670645  ...  9.655792  predictions
                        0.03          437.6  437.945887  ...  9.569740  predictions
                        0.04          433.5  433.856774  ...  9.412411  predictions
            ...                         ...         ...  ...       ...          ...
            1           0.96           34.2   35.883144  ... -0.540094  predictions
                        0.97           38.0   39.549968  ... -0.630391  predictions
                        0.98           42.6   43.395852  ... -0.768797  predictions
                        0.99           46.3   46.764303  ... -0.911782  predictions
                        1.00           50.7   51.073476  ... -1.000000  predictions
            <BLANKLINE>
            [202 rows x 14 columns]

            Get the confidence threshold where the Mean Average Error is the lowest,
            and show the corresponding rows (one per category).

            >>> mae = errors[("absolute", "MAE")]
            >>> mae
            category_id  confidence
            0            0.00          452.0
                         0.01          447.8
                         0.02          442.3
                         0.03          437.6
                         0.04          433.5
                                       ...
            1            0.96           34.2
                         0.97           38.0
                         0.98           42.6
                         0.99           46.3
                         1.00           50.7
            Name: (absolute, MAE), Length: 202, dtype: float64
            >>> best_mae = errors.loc[mae.groupby(level=0).idxmin()]
            >>> best_mae
                                   absolute            ...  relative
                                        MAE      RMSE  ...     q0.75        model
            category_id confidence                     ...
            0           0.89            5.4   7.655064  ...  0.116153  predictions
            1           0.88           10.9  13.939153  ...  0.310000  predictions
            <BLANKLINE>
            [2 rows x 14 columns]
            >>> best_mae.reset_index().iloc[0]
            category_id                     0
            confidence                   0.89
            absolute     MAE              5.4
                         RMSE        7.655064
                         std         7.788881
                         q0.25            0.0
                         q0.50            2.5
                         q0.75            5.5
                         model    predictions
            relative     MRE          0.10748
                         RMSRE       0.145579
                         std         0.145805
                         q0.25            0.0
                         q0.50       0.055717
                         q0.75       0.116153
                         model    predictions
            Name: 0, dtype: object
        """

        def add_image_id(g: list) -> list:
            if "image_id" not in g:
                return [*g, "image_id"]
            return g

        groups = groups_to_list(groups)
        group_names = get_group_names(groups)
        gt_group_dict, *_ = group_relational_data(self.groundtruth, groups, self.images)
        gt_pandas_groups = [gt_group_dict[name] for name in group_names]
        gt_count = (
            self.groundtruth.groupby(add_image_id(gt_pandas_groups))
            .size()
            .rename("gt_count")  # pyright: ignore
        )
        mae_curves = []
        detailed_error_counts = []
        for (
            current_predictions_name,
            current_predictions_frame,
        ) in self.predictions_dictionary.items():
            if confidence_index is None:
                current_confidence_index = np.linspace(0, 1, 101)
            else:
                current_confidence_index = confidence_index

            group_dict, *_ = group_relational_data(
                current_predictions_frame, groups, self.images
            )
            pandas_groups = [group_dict[name] for name in group_names]
            tqdm.pandas()
            prediction_counts = (
                current_predictions_frame.groupby(add_image_id(pandas_groups))[
                    "confidence"
                ]
                .progress_apply(  # pyright: ignore
                    partial(resample_count, new_confidences=current_confidence_index)
                )
                .rename("count")
                .to_frame()
            )
            prediction_counts = prediction_counts.join(gt_count).fillna(0)
            prediction_counts["error"] = (
                prediction_counts["count"] - prediction_counts["gt_count"]
            )
            prediction_counts["rel_error"] = (
                prediction_counts["error"] / prediction_counts["gt_count"]
            )
            prediction_counts["abs_error"] = prediction_counts["error"].abs()
            prediction_counts["abs_rel_error"] = prediction_counts["rel_error"].abs()
            prediction_counts["sq_error"] = prediction_counts["error"].pow(2)
            prediction_counts["sq_rel_error"] = prediction_counts["rel_error"].pow(2)
            prediction_counts["model"] = current_predictions_name
            detailed_error_counts.append(prediction_counts)

            grouped = prediction_counts.groupby([*group_names, "confidence"])
            mae = grouped["abs_error"].mean().rename("MAE")
            rmse = np.sqrt(grouped["sq_error"].mean()).rename("RMSE")
            mre = grouped["abs_rel_error"].mean().rename("MRE")
            rmsre = np.sqrt(grouped["sq_rel_error"].mean().rename("RMSRE"))

            def q_at(y):
                def q(x):
                    return x.quantile(y)

                q.__name__ = f"q{y:0.2f}"
                return q

            stat_agg_functions = ["std", *[q_at(q) for q in quantiles]]
            stats = grouped["error"].agg(stat_agg_functions)
            rel_stats = grouped["rel_error"].agg(stat_agg_functions)
            absolute_result = pd.concat([mae, rmse, stats], axis=1)
            relative_result = pd.concat([mre, rmsre, rel_stats], axis=1)
            absolute_result["model"] = relative_result["model"] = (
                current_predictions_name
            )
            result = pd.concat(
                [absolute_result, relative_result],
                axis=1,
                keys=["absolute", "relative"],
            )
            mae_curves.append(result)
        mae_curves = pd.concat(mae_curves)
        detailed_error_counts = pd.concat(detailed_error_counts)
        return mae_curves, detailed_error_counts

    def compute_normalized_precision_recall(self) -> pd.DataFrame:
        """Compute nAP between detected points and ground truth according to the
        algorithm proposed in [Ref]_

        .. [Ref] Song, Q., Wang, C., Jiang, Z., Wang, Y., Tai, Y., Wang, C. & Wu, Y.
            Rethinking counting and localization in crowds: A purely point-based
            framework.
            2021 IEEE/CVF International Conference on Computer Vision (pp. 3365-3374).
            https://openaccess.thecvf.com/content/ICCV2021/html/Song_Rethinking_Counting_and_Localization_in_Crowds_A_Purely_Point-Based_Framework_ICCV_2021_paper.html
        """
        raise NotImplementedError
