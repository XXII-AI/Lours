from typing import Any

import numpy as np
import pandas as pd
from ot import dist, emd2, sinkhorn2
from scipy.stats import entropy


def earth_mover_distance(
    left: pd.Series,
    right: pd.Series,
    continuous_weights: pd.Series,
    sinkhorn_lambda: float = 0,
) -> float:
    """Compute earth mover distance between two columns of a dataframe.

    Note:
        In the case of ``sinkhorn_lambda`` > 0 this uses the sinkhorn algorithm for a
        faster approximate value.

        See :func:`ot.sinkhorn2`

    Args:
        left: input Series that represents histograms (not necessarily normalized), and
            the index represent the histogram bins
        right: input Series that represents histograms (not necessarily normalized), and
            the index represent the histogram bins. Note that ``left`` and ``right``
            don't necessarily share the same bins.
        continuous_weights: Series of index level names to consider in the
            ``left_right_df`` dataframe for the sinkhorn algorithm.
        sinkhorn_lambda: regularization weight for sinkhorn algorithm. If 0, will use
            literal earth mover distance without regularization (slower but more
            accurate). Defaults to 0.

    Returns:
        distance between the two histograms
    """
    if len(continuous_weights) == 0:
        return 0
    left_sum, right_sum = left.sum(), right.sum()
    if left_sum == 0 or right_sum == 0:
        return float("inf")
    left = left[left > 0]
    right = right[right > 0]
    left_bins = (
        left.index.to_frame()[continuous_weights.index].astype(float)
        * continuous_weights
    ).to_numpy()
    right_bins = (
        right.index.to_frame()[continuous_weights.index].astype(float)
        * continuous_weights
    ).to_numpy()
    distance_matrix = dist(left_bins, right_bins, metric="cityblock")
    distance_matrix = distance_matrix / distance_matrix.max()

    normalized_left = (left / left.sum()).to_numpy()
    normalized_right = (right / right.sum()).to_numpy()
    if sinkhorn_lambda == 0:
        return emd2(
            normalized_left, normalized_right, distance_matrix
        )  # pyright: ignore
    else:
        return sinkhorn2(
            normalized_left, normalized_right, distance_matrix, reg=sinkhorn_lambda
        )  # pyright: ignore


def check_groups(
    histogram: pd.DataFrame | pd.Series,
    category_groups: pd.Series,
    continuous_groups: pd.Series,
) -> None:
    """Check that histogram and groups are well-formed.

    Namely:
     - There should be no overlap between the two groups
     - histogram must have as many index dimensions as the total number of groups
     - histogram multi-index names must be unique
     - there should be a bijection between histogram index names and given category
       and continuous groups


    Args:
        histogram: Series or DataFrame with one or two columns, and a multi index whose
            names must match the next two groups
        category_groups: Series whose index are names of category groups, which should
            be contained in the histogram index
        continuous_groups: Series whose index are names of continuous groups, which
            should be contained in the histogram index

    Raises:
        AssertionError: raises an error when histogram and groups don't respect
            aforementioned criteria
    """
    if len(category_groups) == 0 and len(continuous_groups) == 0:
        raise AssertionError("no group to use pandas' groupby on")
    elif len(category_groups) == 0:
        total_groups = continuous_groups.index
    elif len(continuous_groups) == 0:
        total_groups = category_groups.index
    else:
        total_groups = pd.concat([category_groups, continuous_groups]).index
    assert (
        total_groups.is_unique
    ), "category and continuous groups must have unique and non overlapping values"

    histogram_names = set(histogram.index.names)
    assert len(histogram_names) == len(
        histogram.index.names
    ), "histogram multi index must have index with exclusive names"
    assert histogram_names == set(total_groups), (
        "category and continuous group must be a "
        "perfect partition of the histogram index"
    )
    return


def hist_distance(
    left: pd.Series,
    right: pd.Series,
    category_weights: pd.Series,
    continuous_weights: pd.Series,
    sinkhorn_lambda: float = 0,
) -> float:
    r"""Compute the distance between two distributions described in pandas Series
    representing histograms. Both index must match and may have categorical data or
    continuous data. Distance between categorical data is made with
    Kullback–Leibler divergence and distance between continuous data us made with
    Earth mover distance.

    the distance formula is then

    .. math::
        :label: hist_cost

        D = \sum_{0 \le i < p} \alpha_i KL\left(
            P_{cat, C_i}, Q_{cat, C_i}
            \right) + || \beta || \sum_{i \in \Omega_{cat}} \left(
                P_{cat}(i) \times EMD(P^\beta(i), Q^\beta(i))
            \right)

    where
     - :math:`p \in \mathbb{N}` and :math:`q \in \mathbb{N}` are respectively
       the number of categorical dimensions and continuous dimensions
     - :math:`\Omega_{cat} \subset \mathbb{N}^p` is the set of all possible
       categories, subdivided into :math:`p` dimensions

       .. math::

           \Omega_{cat} &=
               \{ c_{0,0}, c_{1,0} \cdots,  c_{n_0, 0} \}
               \times \cdots
               \times \{ c_{0, p}, \cdots, c_{n_p, p} \} \\
            \Omega_{cat} &=
               C_0 \times \cdots \times C_p

     - :math:`P` is the probability function of the histogram

       .. math::

           P : \begin{array}{lll}
              \Omega_{cat} \times \mathbb{R}^q & \rightarrow & [ 0, 1 ] \\
              (x,y) = (x_0, \cdots, x_p, y_0 \cdots y_p) & \mapsto & P(x,y)
           \end{array}

     - :math:`P_{cat}` is the agglomeration of :math:`P` over continuous dimensions.

       .. math::

           P_{cat} :
               \begin{array}{lll}
                   \Omega_{cat} & \rightarrow & [0, 1] \\
                   x & \mapsto & \iint_{y \in \mathbb{R}^q} P(x, y) dy
                \end{array}

     - :math:`P_{cat, C_i}` is the agglomeration of :math:`P` over continuous
       dimensions and category dimensions except :math:`C_i`

       .. math::

           P_{cat, C_i} &: C_i \rightarrow [0, 1]

           P(x) &= \sum_{
               x' \in C_0 \times \cdots \times C_{i-1} \times C_{i+1}
               \times \cdots \times C_p
           } \iint_{y \in \mathbb{R}^q}
               P(x'_0, \cdots x'_{i-1}, x, x'_{i+1} \cdots x'_p, y) dy

     - :math:`P(x)` is the probability distribution over continuous dimensions for a
       particular category :math:`x \ in \Omega_{cat}`.

       .. math::

            P(x) :
               \begin{array}{lll}
                   \mathbb{R}^q & \rightarrow & [0, 1] \\
                   y & \mapsto & \  P(x, y)
                \end{array}
     - :math:`P^\beta(x)` is the weighted probability distribution over continuous
       dimensions for a particular class :math:`x` and a weight vector :math:`\beta`

       .. math::

           P^\beta(x) &: \mathbb{R}^q \rightarrow [0, 1]

           P^\beta(x,y) &= P \left(x, \frac{\beta}{|| \beta ||} \odot y\right)

     - :math:`\alpha \in \mathbb{R}^p` and
       :math:`\beta \in \mathbb{R}^q` are weight vectors associated to importance
       of each dimensions of :math:`\Omega_{cat} \times \mathbb{R}^q`
     - :math:`\odot` is the Hadamard product

       .. math::

           \beta \odot y = (\beta_j y_j)_{0 \le j < p}
     - :math:`KL` is the Kullback–Leibler divergence
     - :math:`EMD` is the Earth Mover distance

    Note:
        This formula is not symmetric, it is more suited to compare a reference
        distribution (the left one) to a candidate distribution (the right one).


    Args:
        left: pandas Series representing left distribution of probability
            (i.e. the reference)
        right: pandas Series representing left distribution of probability
            (i.e. the candidate)
        category_weights: weights Series vector associated with :math:`\alpha` which is
            applied to the KL divergence (see formula :eq:`hist_cost`). Its index must
            be the names of category groups, that represent ``left`` and ``right``
            indexes dimensions on which to apply KL divergence.
        continuous_weights: weight Series vector associated with :math:`\beta` which is
            applied to the Earth mover's distance (see formula :eq:`hist_cost`).
            Its index must be the names of category groups, that represent
            ``left`` and ``right`` indexes dimensions on which to apply EMD.
        sinkhorn_lambda: regularization term applied to EMV
            (see :func:`earth_mover_distance`). Defaults to 0


    Returns:
        distance between the two multimodal distributions.
    """
    left_right = pd.concat([left, right], axis="columns", keys=["left", "right"])
    check_groups(left_right, category_weights, continuous_weights)
    if 0 in left_right.sum().values:
        return float("inf")

    kl_div = 0
    if len(category_weights) > 0:
        by_cat = left_right.groupby(list(category_weights.index), observed=True)
        by_cat_count = by_cat.sum()

        for axis, weight in category_weights.items():
            assert isinstance(axis, str)
            by_cat_count = left_right.groupby(axis, observed=True).sum()
            axis_entropy = entropy(
                by_cat_count["left"].to_numpy(), by_cat_count["right"].to_numpy()
            )
            kl_div += weight * axis_entropy

        emd_by_cat = by_cat.apply(
            lambda left_right_df: earth_mover_distance(
                left=left_right_df["left"],
                right=left_right_df["right"],
                continuous_weights=continuous_weights,
                sinkhorn_lambda=sinkhorn_lambda,
            )
        )
        emd = (by_cat_count["left"] * emd_by_cat).sum() / by_cat_count["left"].sum()
    else:
        emd = earth_mover_distance(
            left=left_right["left"],
            right=left_right["right"],
            continuous_weights=continuous_weights,
            sinkhorn_lambda=sinkhorn_lambda,
        )
    return (
        np.linalg.norm(category_weights.to_numpy()) * kl_div  # pyright: ignore
        + np.linalg.norm(continuous_weights.to_numpy()) * emd
    )


def df_to_hist(
    data: pd.DataFrame,
    groupby: Any,
    full_index: pd.Index | pd.MultiIndex | None = None,
) -> pd.Series:
    """Convert dataframe to histograms by using pandas'
    `GroupBy <https://pandas.pydata.org/docs/reference/groupby.html>`__ feature

    Args:
        data: DataFrame from which the histogram will be computed. Must have the columns
            specified in groups option.
        groupby: Same ``by`` option for :meth:`pandas.DataFrame.groupby`, will be passed
            directly to ``data.groupby`` method. Can be a mapping, a function, a label,
            or a list of labels.
        full_index: Optional index to reindex the resulting histogram. Useful when some
            value have an occurrence count of 0 and thus don't appear in the induced
            index. Defaults to None.

    Returns:
        pandas Series with multiindex corresponding to the count of occurrences for each
        specified group.
    """
    hist = data.groupby(groupby, observed=False).size()
    hist.name = "histogram"
    if full_index is None:
        if isinstance(hist.index, pd.MultiIndex):
            full_index = pd.MultiIndex.from_product(hist.index.levels)
        else:
            full_index = hist.index
    return hist.reindex(full_index).fillna(0)


def dataset_share_distance(left_share: pd.Series, right_share: pd.Series) -> float:
    r"""Compute the distance between two dataset share histograms (where bins are
    splits) by using Intersection over Union (IoU). We use this distance instead of KL
    because we don't want an infinite distance when one of the split is empty.

    .. math::
        D = \frac{\sum_{i=0}^{n_{splits}} min(left(i), right(i))}
                  {\sum_{i=0}^{n_{splits}} max(left(i), right(i))}

    Args:
        left_share: Series representing target histogram of split sizes.
            It has to be normalized.
        right_share: candidate histogram of split sizes

    Returns:
        distance computed
    """
    left_share = left_share / left_share.sum()
    right_share = right_share / right_share.sum()
    intersection = np.minimum(left_share, right_share)
    union = np.maximum(left_share, right_share)
    if union.sum() == 0:
        print(left_share, right_share)
        raise ValueError()
    return 1 - intersection.sum() / union.sum()
