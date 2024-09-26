import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn

from ...utils import BBOX_COLUMN_NAMES


def construct_matches_df(
    predictions_targets_df: pd.DataFrame, min_iou: float = 0
) -> pd.DataFrame:
    """From a dataframe with targets and predictions, all concatenated together,
    construct a list of match pairs between prediction and targets. Unmatched
    predictions or targets get a <NA> match id. Note that all bounding boxes in the
    input dataframe are assumed to be of the same category and the same image, the
    grouping must have already been done by the user before.

    Args:
        predictions_targets_df: DataFrame comprising target and prediction info must
            have the following columns:

             - ``groundtruth`` : bool value to know if it's a target or a prediction
             - ``box_x_min``, ``box_y_min``, ``box_width``, ``box_height``: Bounding box
               information to compute IoU

        min_iou: IoU above which the detection is considered valid.
            Note that the lower bound is not inclusive. Defaults to 0.

    Returns:
        DataFrame of matches. Will contain prediction_id and groundtruth_id columns.
        Index is irrelevant. Each prediction id and target id should appear once and
        only once. As such, at worse (no match at all), the dataframe will be N+M rows
        with N the number of predictions and M the number of targets

    """
    groundtruth = predictions_targets_df[predictions_targets_df["groundtruth"]]
    predictions = predictions_targets_df[~predictions_targets_df["groundtruth"]]
    ious = get_ious(groundtruth, predictions)
    detection_matches, groundtruth_matches = get_matches(
        ious, predictions["confidence"], min_iou
    )
    matches = detection_matches.reset_index(names="prediction_id").rename(
        columns={"match_id": "groundtruth_id"}
    )
    not_detected = groundtruth_matches[groundtruth_matches["match_id"].isna()]
    not_detected = not_detected.reset_index(names="groundtruth_id").rename(
        columns={"match_id": "prediction_id"}
    )
    return pd.concat([matches, not_detected])


def get_ious(groundtruth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """From two dataframes of annotations, generate a matrix of iou of size N x M where
    N is the number of predictions and M is the number targets.
    Rows are sorted by prediction confidence

    Note that this does not check the category_id, only the bounding box coordinates.

    Next, encapsulate it in a dataframe with index and columns named after prediction
    and target ids.

    Args:
        groundtruth: DataFrame comprising bounding box targets data.
            Must include at least ``box_x_min``, ``box_y_min``, ``box_width``,
            ``box_height``
        predictions: DataFrame comprising bounding box prediction data.
            Must include same columns as groundtruth, plus the ``confidence`` column.

    Returns:
        DataFrame comprising iou values between groundtruth and predictions. Index is
        prediction id, column name is target id
    """
    # Extract bbox coordinates from groundtruth and pred. Note that prediction bbox data
    # is one more dimension to use array broadcasting
    # each array is of shape [M] (implicitly, [1, M])
    x1, y1, w1, h1 = groundtruth[BBOX_COLUMN_NAMES].values.T
    # each array is of shape [N, 1]
    x2, y2, w2, h2 = predictions[BBOX_COLUMN_NAMES].values.T[..., None]
    # Compute area of intersection
    # Here we use array broadcasting so that every constructed array is of size NxM
    xmin = np.maximum(x1, x2)
    xmax = np.minimum(x1 + w1, x2 + w2)
    ymin = np.maximum(y1, y2)
    ymax = np.minimum(y1 + h1, y2 + h2)
    area = (xmax - xmin) * (ymax - ymin)
    area[(xmax < xmin) | (ymax < ymin)] = 0
    ious = pd.DataFrame(
        area / (w1 * h1 + w2 * h2 - area),
        index=predictions.index,
        columns=groundtruth.index,
    )

    return ious


def get_matches(
    iou_df: pd.DataFrame, confidence: pd.Series | None = None, min_iou: float = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the best matching target for every prediction and
    return matching target (if any) for every prediction and
    matching prediction (if any) for every target
    Prediction are either reordered by confidence, or assumed already ordered in the
    first place.


    Args:
        iou_df: IoU values matrix encapsulated in a dataframe to index
            rows with prediction ids and columns with target ids
        confidence: series with the number of rows as iou_df, will
            be used to reorder iou_df's rows in descending order. If not given, will
            assume iou_df is already ordered.
        min_iou: Minimum IoU value above which a match is considered
            valid.

    Returns:
        dataframes of matching ids with corresponding
        ious. First df is indexed by prediction ids, second df is indexed by target id
    """
    if confidence is not None:
        ious = iou_df.reindex(confidence.sort_values(ascending=False).index)
    else:
        ious = iou_df.copy()
    # Note that we use the Int64 type, which is the regular int64 + NA value, which is
    # used here to designate the absence of match
    # Both matches dataframes are initialized to have zero match and will be iteratively
    # updated
    detection_matches = pd.DataFrame(
        np.zeros((len(ious), 2)),
        index=ious.index,
        columns=["iou", "match_id"],
    )
    groundtruth_matches = pd.DataFrame(
        np.zeros((len(ious.columns), 2)),
        index=ious.columns,
        columns=["iou", "match_id"],
    )
    detection_matches["match_id"] = pd.NA
    groundtruth_matches["match_id"] = pd.NA

    match_dtypes = {"iou": float, "match_id": "Int64"}
    detection_matches = detection_matches.astype(match_dtypes)
    groundtruth_matches = groundtruth_matches.astype(match_dtypes)

    # Iterative vectorize matching algorithm
    # 1 - Get best target match of each prediction
    # 2 - Remove every prediction and corresponding target until the first duplicate
    # 3 - Update aforementioned match dataframes accordingly
    # 4 - Repeat with this new subset
    # Note that we don't need to compute best target match each time (only until the
    # first duplicate), but that fact that it is vectorized across the iou matrix
    # makes it basically free.
    while len(ious) > 0:
        best_iou = ious.max(axis=1)
        valid = best_iou > min_iou
        ious = ious[valid]
        best_iou = best_iou[valid]
        if len(ious) == 0:
            break
        best_matches = ious.idxmax(axis=1)
        duplicated = best_matches.duplicated()
        if not duplicated.max():
            # No duplicate (max is False), perfect matching
            first_duplicated = len(duplicated)
        else:
            # Get first occurrence of duplicated == True
            first_duplicated = duplicated.argmax()

        # Partition between matched and not matched yet
        matched = best_matches.iloc[:first_duplicated]
        matched_iou = best_iou.iloc[:first_duplicated]
        not_matched = best_matches.iloc[first_duplicated:]

        ious = ious.loc[not_matched.index].drop(pd.Index(matched.values), axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            detection_matches.loc[matched.index, "match_id"] = matched
            groundtruth_matches.loc[matched, "match_id"] = matched.index.to_numpy()

        # Get corresponding iou values
        detection_matches.loc[matched.index, "iou"] = matched_iou
        groundtruth_matches.loc[matched, "iou"] = matched_iou

    return detection_matches, groundtruth_matches


def pr_curve(
    results: pd.DataFrame,
    min_iou: float = 0,
    betas: Iterable[float] = (1,),
    reindex_series: pd.Series | None = None,
) -> pd.DataFrame:
    r"""Construct Precision Recall curve from results dataframe and minimum iou below
    which detection is considered invalid

    Additionally, computes F-score with different :math:`\beta` values with the
    following equation.

    .. math:
        F_\beta = \frac{(1 + \beta^2)(\text{precision} \times \text{recall})}
                       {\text{precision} \times \beta^2 + \text{recall}}

    Args:
        results: Dataframe modelling detections, with corresponding
            confidence and groundtruth (whether this detection would be True positive or
            a False positive). Should include the columns ``groundtruth``, ``iou`` and
            ``confidence``, and rows should be sorted so that confidence values are
            sorted.
        min_iou: Value below which the detection is considered
            invalid. In other words, the groundtruth becomes ``False``. The prediction
            becomes a False Positive, and the corresponding groundtruth is a False
            negative. Defaults to 0.
        betas: beta values to compute the F-Score with. Must be an iterable of floats.
            Defaults to ``(1,)``
        reindex_series: Recall bins to reindex the curve. before returning it.

    Returns:
        Precision Recall curve dataframe. Columns are ``precision``,
        ``recall``, ``f{beta}_score`` and ``confidence_threshold``, where betas are the
        given :math:`\beta` values in ``betas`` (see equation above).
        Index is irrelevant.
    """
    results = results.sort_values("confidence", ascending=False)
    ntargets = results["groundtruth"].sum()
    confidence = results["confidence"].to_numpy()
    distinct_value_indices = np.diff(confidence).astype(bool)
    distinct_value_indices = np.append(distinct_value_indices, True)
    confidence = confidence[distinct_value_indices]
    # Cumulative sum of true positives, from which we only extract the maximum for
    # distinct confidence value
    tp_count = (results["groundtruth"] * (results["iou"] > min_iou)).to_numpy().cumsum()
    tp_count = tp_count[distinct_value_indices]

    # Precision and recall
    # Precision is true positive / number of positive predictions
    # Recall is true positive / number of total targets (even the ones with IOU of zero)
    precision = tp_count / (1 + distinct_value_indices.nonzero()[0])
    # In the degenerate case of no targets to be detected, the recall cannot be computed
    # Hence the NaN
    recall = tp_count / ntargets if ntargets > 0 else tp_count * np.nan

    # Add 2 points for each extreme
    # Precision will not be above first value,
    # no matter how  high the confidence threshold is
    # Recall will not be above last value,
    # no matter how low the confidence threshold is
    # We still add the extremal points with recall = 1 and precision = 0 and
    # precision = 1 and recall = 0 for completeness
    # Wen this curve is reindexed by precision or recall
    # (which is the case for pycocotools).

    # Note pyright ignore flags to be removed as soon as we get the pre-commit hook
    # pyright 1.1.206
    # See https://github.com/microsoft/pyright/issues/2809
    precision = np.concatenate([[1], precision[:1], precision, [0, 0]])
    recall = np.concatenate([[0, 0], recall, recall[-1:], [1]])
    confidence = np.concatenate([[1, 1], confidence, [0, 0]])

    # Make sure the precision is only decreasing.
    # The rationale is that the true precision recall curve (thus with infinite number
    # of points) is only decreasing.
    # But the way it is constructed with a finite dataset makes precision drop when
    # A false positive occurs, and increase again at the next true positive.
    # Most conservative way of constructing a realistic curve is to make points of
    # dropping precision equal to the next highest precision.
    # For that we use numpy's universal function, and more specifically the accumulate
    # feature
    # see https://numpy.org/doc/stable/reference/generated/numpy.ufunc.accumulate.html
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    f_scores = {}
    for beta in betas:
        # See https://en.wikipedia.org/wiki/F-score for formula
        f_scores[f"f{beta}_score"] = (
            (1 + beta**2) * (precision * recall) / (precision * beta**2 + recall + 1e-5)
        )
    result = pd.DataFrame(
        np.stack([precision, recall, confidence, *f_scores.values()], axis=1),
        columns=["precision", "recall", "confidence_threshold", *f_scores.keys()],
    ).fillna(0)

    # Remove points which are not useful to draw the curve or compute the mAP, ie the
    # points  that are on a straight line
    constant_precision = (result["precision"].diff() == 0) & (
        result["precision"].diff(-1) == 0
    )
    result = result[~constant_precision]

    constant_recall = (result["recall"].diff() == 0) & (result["recall"].diff(-1) == 0)
    result = result[~constant_recall]

    if reindex_series is not None:
        result = result.set_index(reindex_series.name)
        # Remove duplicated index values, otherwise reindex will error
        result = result[~result.index.duplicated(keep="last")]
        result = result.fillna(0)
        result = result.reindex(reindex_series, method="ffill").reset_index()
    return result


def compute_average_precision(pr_curve: pd.DataFrame) -> float:
    """Compute average precision from dataframe with precision and recall values.
    Precision values are averaged over recall values.

    Note:
        We compute the right Riemann sum, i.e. we only consider the value on the right
        for a particular recall interval.

    Args:
        pr_curve: Dataframe with ``precision`` and ``recall`` columns.

    Returns:
        Average precision for this particular PR curve
    """
    sorted_pr_curve = pr_curve.sort_values("recall")
    precision = sorted_pr_curve["precision"]
    # First value of recall_diff is NaN, replace it with 0 so that we discard
    # the first precision value
    recall_diff = sorted_pr_curve["recall"].diff().fillna(0)
    return (precision * recall_diff).sum()


def resample_count(
    original_confidences: Iterable[float], new_confidences: Iterable[float]
) -> pd.Series:
    """Take a sequence of confidence values and resample it assuming at each new
    original confdience value, one object is added.

    Result is the number of objects that would have been detected for each value in
    new confidence.

    Note:
        ``new_confidences`` must be sorted unique values.

    Args:
        original_confidences: Original set of confidence value. Each confidence value
            corresponds to one detected object.
        new_confidences: New set of confidence values to resample the number of detected
            objects from. Usually, a range of N elements, from 0 to 1.

    Returns:
        Series named ``count`` with the same length as ``new_confidences``, index set as
        ``new_confidences``, named ``confidence``, and values set to count values
        corresponding to confidence threshold given in the index.
    """
    counts = (
        pd.Series(list(original_confidences))
        .value_counts()
        .sort_index(ascending=False)
        .cumsum()
    )
    new_confidences = pd.Index(new_confidences, name="confidence")
    resampled = counts.reindex(new_confidences, method="ffill").fillna(0)

    return resampled


def confusion_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute the confusion matrix for a given DataFrame.

    Args:
        matches: DataFrame containing the matches between groundtruth and predictions
            in which we expect to have the following columns :

            - ``prediction_label``
            - ``groundtruth_label``

            corresponding to the predicted and groundtruth labels,
            respectively, in order to compute the confusion matrix.

    Returns:
        A confusion matrix as DataFrame with class names as column names and row ids.
    """
    y_pred = matches["prediction_label"].fillna("None").astype(str)
    y_true = matches["groundtruth_label"].fillna("None").astype(str)

    # Create a list of all possible classes
    all_classes = sorted(set(y_pred) | set(y_true))
    if "None" in all_classes:
        all_classes.remove("None")
        all_classes.append("None")

    cm = confusion_matrix_sklearn(y_true, y_pred, labels=all_classes, normalize="true")

    return pd.DataFrame(
        cm, index=pd.Index(all_classes, name="label"), columns=all_classes
    )


def display_confusion_matrix(confusion_matrix: pd.DataFrame, title: str = ""):
    """Display a ConfusionMatrixDisplay object for a given Dataframe.

    Args:
        confusion_matrix: Dataframe containing the confusion matrix data
            as computed by :func:`.confusion_matrix`
        title: Confusion matrix's title

    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError as e:
        raise ImportError(
            "Plotting libraries could not be loaded, make sure you have installed"
            " Lours with the 'plot-utils' extra"
        ) from e

    display_labels = confusion_matrix.columns.to_list()

    # scaling text inside the matrix cells.
    # Text is not scaled according to the number of labels.
    # We need to make the text smaller if the matrix cells are getting smaller as well
    # somehow, the size of the cell is both inversly proportional to the number of
    # labels, and also gets smaller if the longest label is very long.
    # this algorithm tries to find the right font size, from xx-small to regular
    text_kw = {}
    n_labels = len(display_labels)
    max_label_length = max(map(len, display_labels))
    n_labels += max_label_length / 3
    if n_labels > 15:
        text_kw["fontsize"] = "xx-small"
    elif n_labels > 11:
        text_kw["fontsize"] = "x-small"
    elif n_labels > 9:
        text_kw["fontsize"] = "small"

    plot = ConfusionMatrixDisplay(
        confusion_matrix.values.round(2),
        display_labels=display_labels,
    ).plot(text_kw=text_kw)
    # Use maptlotlib's tick labels function so that we can rotation around the tick
    # and not around the label center. Otherwise, long labels might end up overlapping
    # the next labels
    plot.ax_.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize="small" if max_label_length > 10 else "medium",
    )
    if max_label_length > 10:
        plot.ax_.set_yticklabels(
            display_labels,
            fontsize="small",
        )
    plt.title(title)
    plt.xlabel("Prediction label")
    plt.ylabel("Grounthruth label")
    plt.tight_layout()
