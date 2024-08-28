from collections.abc import Callable, Iterable
from random import seed, shuffle
from typing import overload
from warnings import warn

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ...utils.grouper import (
    ContinuousGroup,
    get_group_names,
    group_list,
    group_relational_data,
    groups_to_list,
)
from .balanced_groups import dataset_share_distance, df_to_hist, hist_distance
from .disjoint_groups import make_atomic_chunks


def get_winner(
    split_hists: pd.DataFrame | None,
    split_hists_distances: pd.Series | None,
    candidate_hist: pd.Series | None,
    split_sizes: pd.Series,
    candidate_size: int,
    hist_cost_function: Callable[[pd.Series], float],
    share_cost_function: Callable[[pd.Series], float],
    hist_cost_weight: float = 1,
    share_cost_weight: float = 1,
) -> tuple[str, pd.DataFrame | None, pd.Series | None, pd.Series]:
    """Get the best split i.e. with the lowest from series of precomputed costs.
    The series are histogram costs, i.e. with distribution distances for values the
    user which to be evenly distributed between splits, and the share costs the IOU
    distance between

    The result is then the key of the dictionary with the lowest consolidated cost.
    A special case is when all distribution costs are infinite. In that case, only
    consider the share cost.

    Args:
        split_hists: DataFrame containing the current histograms of splits. Columns are
            splits, and rows are histogram bins
        split_hists_distances: Series containing the cached distance values of distance
            between the split hist and the target histogram. If set to None, will
            recompute them
        candidate_hist: Series containing the histogram of the candidate atom. rows are
            the same as ``split_hists``
        split_sizes: Series containing the sizes of each split each row is a split.
        candidate_size: size of current atom. Depending on how the split is done, it's
            not necessary the same as the sum of candidate histogram.
        hist_cost_function: function that computes a score for a dataframe of
            histograms. This will be used to compute the histogram cost for each split
            if the atom was to be assigned to it.
        share_cost_function: function that computes a score for dataset repartition
            against a target split share. This is used to compute the cost of assigning
            the candidate atom to each split.
        hist_cost_weight: weight applied to histogram cost to choose the winner split.
            The higher, the more important the histogram cost will be for the decision.
            Defaults to 1.
        share_cost_weight: weight applied to share cost to choose the winner split.
            The higher, the more important the share cost will be for the decision.
            Defaults to 1.


    Returns:
        A tuple with 4 elements
         - name of the winning split
         - updated split histograms, as a DataFrame similar to ``split_hists``
           (None if given ``split_hists`` was None)
         - updated split hist costs, as a Series, similar to ``split_hists_distances``
           (None if given ``split_hists`` was None)
         - updated share of splits, as a Series, similar to ``split_shares``
    """
    # Construct aggregated split histogram cost Series, where each row is the split
    # we could assign the new atom, and the value is the corresponding cost that
    # We try to minimize
    split_names = split_sizes.index
    if split_hists is not None:
        assert candidate_hist is not None
        if split_hists_distances is None:
            split_hists_distances = split_hists.apply(hist_cost_function)
        updated_split_hists = split_hists.add(candidate_hist, axis="index")
        updated_split_hist_distances = updated_split_hists.apply(hist_cost_function)
        split_hist_distance_square = split_hists_distances.values
        ones = np.ones_like(split_hist_distance_square, dtype=int)
        split_hist_distance_square = split_hist_distance_square[:, None] @ ones[None]
        split_hist_distance_square[ones, ones] = updated_split_hist_distances
        aggregated_split_hists_costs = pd.Series(
            split_hist_distance_square.sum(axis=0), index=split_names
        )
    else:
        split_hists_distances = None
        updated_split_hist_distances = None
        updated_split_hists = None
        aggregated_split_hists_costs = pd.Series(0, index=split_names)

    # Construct aggregated share cost Series, whereas above, rows are split and
    # values are IoU between target share and the resulting share
    # if the atom was to be assigned to that split
    updated_sizes = split_sizes + candidate_size
    split_size_square = pd.DataFrame(
        split_sizes.values + np.diag(updated_sizes - split_sizes),
        index=split_names,
        columns=split_names,
    )

    aggregated_share_costs = split_size_square.apply(share_cost_function, axis=1)
    assert isinstance(aggregated_share_costs, pd.Series)

    infinite_histogram_cost = aggregated_split_hists_costs == float("inf")
    if all(infinite_histogram_cost):
        consolidated_cost = aggregated_share_costs
    else:
        consolidated_cost = (
            hist_cost_weight * aggregated_split_hists_costs
            + share_cost_weight * aggregated_share_costs
        )
    winner = consolidated_cost.idxmin()
    assert isinstance(winner, str)

    if split_hists is not None:
        split_hists[winner] = updated_split_hists[winner]  # pyright: ignore
        split_hists_distances[winner] = updated_split_hist_distances[  # pyright: ignore
            winner  # pyright: ignore
        ]

    split_sizes[winner] = updated_sizes[winner]
    return winner, split_hists, split_hists_distances, split_sizes


@overload
def split_dataframe(
    input_data: pd.DataFrame,
    root_data: pd.DataFrame,
    key_to_root: str = "image_id",
    input_seed: int = 0,
    split_names: Iterable[str] = ("train", "valid"),
    target_split_shares: Iterable[float] = (0.8, 0.2),
    split_column_name: str = "split",
    keep_separate_groups: group_list = ("image_id",),
    keep_balanced_groups: group_list = ("category_id",),
    keep_balanced_groups_weights: Iterable[float] | None = None,
    inplace: bool = False,
    split_at_root_level: bool = False,
    hist_cost_weight: float = 1,
    share_cost_weight: float = 1,
    earth_mover_regularization: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pass


@overload
def split_dataframe(
    input_data: pd.DataFrame,
    root_data: None = None,
    key_to_root: str = "image_id",
    input_seed: int = 0,
    split_names: Iterable[str] = ("train", "valid"),
    target_split_shares: Iterable[float] = (0.8, 0.2),
    split_column_name: str = "split",
    keep_separate_groups: group_list = ("image_id",),
    keep_balanced_groups: group_list = ("category_id",),
    keep_balanced_groups_weights: Iterable[float] | None = None,
    inplace: bool = False,
    split_at_root_level: bool = False,
    hist_cost_weight: float = 1,
    share_cost_weight: float = 1,
    earth_mover_regularization: float = 0,
) -> pd.DataFrame:
    pass


@overload
def split_dataframe(
    input_data: pd.DataFrame,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
    input_seed: int = 0,
    split_names: Iterable[str] = ("train", "valid"),
    target_split_shares: Iterable[float] = (0.8, 0.2),
    split_column_name: str = "split",
    keep_separate_groups: group_list = ("image_id",),
    keep_balanced_groups: group_list = ("category_id",),
    keep_balanced_groups_weights: Iterable[float] | None = None,
    inplace: bool = False,
    split_at_root_level: bool = False,
    hist_cost_weight: float = 1,
    share_cost_weight: float = 1,
    earth_mover_regularization: float = 0,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    pass


def split_dataframe(
    input_data: pd.DataFrame,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
    input_seed: int = 0,
    split_names: Iterable[str] = ("train", "valid"),
    target_split_shares: Iterable[float] = (0.8, 0.2),
    split_column_name: str = "split",
    keep_separate_groups: group_list = ("image_id",),
    keep_balanced_groups: group_list = ("category_id",),
    keep_balanced_groups_weights: Iterable[float] | None = None,
    inplace: bool = False,
    split_at_root_level: bool = False,
    hist_cost_weight: float = 1,
    share_cost_weight: float = 1,
    earth_mover_regularization: float = 0,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Perform the split operation on input_data and root_data.

    This algorithm works in 2 steps:

    1. divide the dataframe into atomic sub frames. Given the image and annotation
        attributes that need to be kept separate, we can construct sub frame of elements
        that cannot be in different splits.
    2. Construct the split dataframes iteratively by trying to keep given column values
        with a balanced repartition between splits, along with keeping split sizes as
        close to target share as possible. Each atomic sub frame is routed to the split
        that minimize a cost function which try to optimize repartition targets.

    Args:
        input_data: DataFrame containing input_data information, must contain at least
            the column given in ``key_to_root``.
        root_data: DataFrame containing image information. its index must contain all
            values contained in the ``image_id`` column of the input_data DataFrame.
        key_to_root: name of the column in input that refers to id in root data
            dataframe. Defaults to "image_id".
        input_seed: Seed used for shuffling sub frames before beginning step 2 of
            splitting algorithm. Defaults to 0.
        split_names: Names of splits. Must be the same length as
            ``target_split_shares``. Defaults to ("train", "valid").
        target_split_shares: List of relative size of each split. Must be the same
            length as ``split_names``, and will be normalized so that its sum is 1.
            Defaults to (0.8, 0.2).
        split_column_name: Name of the column where the split value of dataset will be
            read and written. Defaults to "split".
        keep_separate_groups: columns in ``input_data`` or `root_data``
            DataFrame to keep separate. That is for a particular column, two rows with
            the same value cannot be in different splits. Defaults to ("image_id",).
        keep_balanced_groups: columns or groups (as defined in
            ``input_data`` or ``root_data`` DataFrames to keep balanced.
            That is for a particular column, the distribution of values is the same
            between original DataFrame and its split, as much as possible.
            Defaults to ("category_id",).
        keep_balanced_groups_weights: Importance of each group to keep balanced when
            computing histogram cost. If not None, must be of the same size as
            ``keep_separate_groups``. Defaults to None.
        inplace: If set, will modify dataframes inplace. This can silently modify some
            objects (like Datasets) that use them. Defaults to False.
        split_at_root_level: If set, will compute split sizes (and thus share distances)
            at root level, i.e. regarding sizes in the ``root_data`` dataframe. As a
            consequence, the split column name will be added to
            ``keep_separate_input_groups`` if it's not already in it, and the number of
            rows in the input data per row in root data will not have any influence on
            the share cost.
        hist_cost_weight: importance of histogram cost for balanced groups.
            The higher, the more important the histogram cost will be for the decisio of
            where to put each split. Defaults to 1.
        share_cost_weight: importance of share cost for balanced groups.
            The higher, the more important the share cost will be for the decision of
            where to put each split. Defaults to 1.
        earth_mover_regularization: Regularization parameter applied to sinkhorn's
            algorithm during earth mover distance computation. See
            :func:`.earth_mover_distance`. Defaults to 0

    Returns:
        new annotation and root_data with the split column populated with the
        corresponding split name.
    """
    target_split_shares = pd.Series(list(target_split_shares), index=[*split_names])
    target_split_shares /= target_split_shares.sum()

    keep_balanced_groups = groups_to_list(keep_balanced_groups)
    for g in keep_balanced_groups:
        if isinstance(g, ContinuousGroup) and g.label_type == "intervals":
            g.label_type = "mid"
    keep_balanced_group_names = get_group_names(keep_balanced_groups)
    keep_separate_groups = groups_to_list(keep_separate_groups)

    if keep_balanced_groups_weights is None:
        keep_balanced_groups_weights = pd.Series(
            1, index=keep_balanced_group_names, dtype=float
        )
    else:
        keep_balanced_groups_weights = pd.Series(
            list(keep_balanced_groups_weights),
            index=keep_balanced_group_names,
            dtype=float,
        )

    if not inplace:
        input_data = input_data.copy()
        root_data = root_data.copy() if root_data is not None else None

    if split_at_root_level:
        assert key_to_root in input_data.columns
    if root_data is not None:
        assert key_to_root in input_data.columns
        assert input_data[key_to_root].isin(root_data.index).all()
        if split_column_name in root_data.columns:
            input_data[split_column_name] = root_data.loc[
                input_data[key_to_root], split_column_name
            ].values
        if split_at_root_level and key_to_root not in keep_separate_groups:
            keep_separate_groups.append(key_to_root)

    if split_column_name in input_data:
        already_assigned = input_data[split_column_name].isin(split_names)
        if already_assigned.sum() == len(input_data):
            warn(
                "Every row in the DataFrame is already assigned to a given split. It's"
                " possible you forgot to remove the already existing split values"
                " before splitting ",
                RuntimeWarning,
            )

    keep_separate_input_groups_dict, *_ = group_relational_data(
        input_data,
        keep_separate_groups,
        root_data,
        key_to_root=key_to_root,
    )
    keep_separate_input_pandas_groups = list(keep_separate_input_groups_dict.values())

    print("Separating input data into atomic chunks")
    atomic_chunks, assigned_chunks = make_atomic_chunks(
        data=input_data,
        groups=keep_separate_input_pandas_groups,
        split_column=split_column_name,
        split_names=target_split_shares.index,  # pyright: ignore
    )
    if not atomic_chunks:
        print("No chunk to distribute")
    else:
        print(
            f"{len(atomic_chunks)} chunks to distribute"
            f" across {len(target_split_shares)} splits"
        )

    # Construct a split dictionary, containing the indexes of input_data, belonging
    # to each split
    splits: dict[str, list] = {str(name): [] for name in target_split_shares.index}
    split_sizes = pd.Series(0, index=target_split_shares.index)

    def share_cost_function(candidate_shares: pd.Series) -> float:
        return dataset_share_distance(target_split_shares, candidate_shares)

    (
        keep_balanced_groups_dict,
        category_groups,
        continuous_groups,
    ) = group_relational_data(
        input_data,
        keep_balanced_groups,
        root_data,
        key_to_root=key_to_root,
    )

    category_weights = keep_balanced_groups_weights[
        keep_balanced_groups_weights.index.isin(category_groups)
    ]
    continuous_weights = keep_balanced_groups_weights[
        keep_balanced_groups_weights.index.isin(continuous_groups)
    ]

    keep_balanced_pandas_groups = list(keep_balanced_groups_dict.values())
    if keep_balanced_pandas_groups:
        target_hist = df_to_hist(input_data, keep_balanced_pandas_groups)
        # Construct the histogram of each split that will be updated along with
        # their construction. split_hists is a dataframe of histograms.
        # Each column is a split
        split_hists = pd.DataFrame(
            0, index=target_hist.index, columns=target_split_shares.index
        )

        # Function that we will apply to the split histogram dataframe
        def hist_cost_function(split_hist: pd.Series) -> float:
            return hist_distance(
                target_hist,
                split_hist,
                category_weights,
                continuous_weights,
                sinkhorn_lambda=earth_mover_regularization,
            )

    else:
        target_hist = None
        split_hists = None

        def hist_cost_function(split_hist: pd.Series) -> float:
            return 0.0

    # Already assigned chunks belong to a split, so we add them to the split's index
    # list and update their histogram
    for name, group in assigned_chunks.items():
        if name in split_names:
            splits[name].append(group.index)
            if target_hist is not None:
                assert split_hists is not None
                split_hists[name] += df_to_hist(
                    group,
                    keep_balanced_pandas_groups,
                    full_index=target_hist.index,
                )
            split_sizes[name] += len(group)
        else:
            atomic_chunks.append(group)

    seed(input_seed)
    shuffle(atomic_chunks)

    if target_hist is not None:
        assert split_hists is not None
        candidate_hists = [
            df_to_hist(
                atom,
                keep_balanced_pandas_groups,
                full_index=target_hist.index,
            )
            for atom in atomic_chunks
        ]
        # Construct the distances between each split histogram and the target histogram
        # If the splits are not populated, we should get infinity everywhere
        split_hists_distances = split_hists.apply(hist_cost_function)
    else:
        candidate_hists = None
        split_hists_distances = None
    # For each atom, decide where it will go by computing the overall distance between
    # each split histogram and the target histogram, under the hypothesis of going
    # to a particular split. The winner split is the hypothesis with the lowest score,
    # gets the atom and its histogram gets updated
    for i, atom in enumerate(tqdm(atomic_chunks, disable=len(atomic_chunks) == 0)):
        if len(atom) == 0:
            continue
        if candidate_hists is not None:
            current_candidate_hist = candidate_hists[i]
        else:
            current_candidate_hist = None
        if split_at_root_level:
            candidate_size = len(atom[key_to_root].unique())
        else:
            candidate_size = len(atom)
        winner, split_hists, split_hists_distances, split_sizes = get_winner(
            split_hists,
            split_hists_distances,
            current_candidate_hist,
            split_sizes,
            candidate_size,
            hist_cost_function,
            share_cost_function,
            hist_cost_weight,
            share_cost_weight,
        )
        splits[winner].append(atom.index)

    for name, split in splits.items():
        if not split:
            continue
        input_data_to_mark = np.concatenate(split)
        input_data.loc[input_data_to_mark, split_column_name] = name
        if root_data is not None and split_at_root_level:
            root_data_to_mark = input_data.loc[
                input_data[split_column_name] == name, key_to_root
            ].unique()
            root_data.loc[root_data_to_mark, split_column_name] = name

    if root_data is not None:
        return input_data, root_data
    else:
        return input_data
