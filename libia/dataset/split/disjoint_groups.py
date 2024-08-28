import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass
class IndexedSet:
    """Class representing a set with a corresponding list of indexes indicating what
    sets in the initial list where used to construct this one. In other word, there's
    an original list of sets and the union of all sets indexed make up the current
    set.
    """

    index: set[int]
    """index $i$ of sets $S_i$ that were used when constructing this set.
    """

    merged_set: set
    r"""
    Resulting set.

    .. math::
        S = \bigcup_{i \in \text{index}} S_i
    """

    def union(self, *others: "IndexedSet") -> "IndexedSet":
        r"""Perform the union operation. union operation is applied on both index sets
        and the sets themselves.

        Args:
            *others: Iterable of $n$ other indexed sets :math:`(S_i, \text{index}_i)`
                to perform the union operation

        Returns:
            new indexed set with

            .. math::

                \text{index} &= \text{index}_1 \cup \text{index}_2 \cup \cdots
                  \cup \text{index_n} \\
                S &= S_1 \cup S_2 \cup \cdots \cup S_n
        """
        return IndexedSet(
            self.index.union(*[o.index for o in others]),
            self.merged_set.union(*[o.merged_set for o in others]),
        )

    def is_disjoint(self, other: "IndexedSet") -> bool:
        """Tell if the intersection between current index set and another one is empty
        or not

        Args:
            other: other indexed set that we want the intersection with

        Returns:
            True if intersection is empty, False otherwise
        """
        return self.merged_set.intersection(other.merged_set) == set()


def factorize_sets(input_sets: Sequence[set]) -> list[list[int]]:
    r"""From an index-able sequence of sets, partition all possible values in factor
    sets so that two elements in a particular factor set can be linked with a sequence
    of input sets with a non-null intersection.


    .. math::

        \widehat{S} = \bigcup_i S_i \in input sets

        \forall x,y \in \widehat{S} , \exists i_0 , i_1, \cdots , i_n,
          x \in S_{i_0}, y \in S_{i_n}, \forall j,
          S_{i_j} \cap S_{i_{j+1}} \neq \emptyset


    Args:
        input_sets: sequence of sets with possible overlapping values that need to be
            factorized.


    Returns:
       list of set indices for each factor. That is, the index in the input sets
       sequence to recreate the factor sets with a union operation.
    """
    indexed_input_sets = [
        IndexedSet({i}, current_set) for i, current_set in enumerate(input_sets)
    ]
    merged_indices = []
    while indexed_input_sets:
        first_id_set, *remaining = indexed_input_sets
        to_merge, to_keep = [], []
        for id_set in remaining:
            (
                to_keep.append(id_set)
                if first_id_set.is_disjoint(id_set)
                else to_merge.append(id_set)
            )
        if not to_merge:
            merged_indices.append(first_id_set.index)
            indexed_input_sets = remaining
            continue
        indexed_input_sets = [first_id_set.union(*to_merge)] + to_keep
    return merged_indices


def give_already_assigned(
    data: pd.DataFrame, split_column: str = "split", split_names: Iterable[str] = ()
) -> tuple[list[pd.DataFrame], dict[str, pd.DataFrame]]:
    """Divide a DataFrame with a split column into chunks with an assigned split and
    unassigned chunks. Unassigned chunks are chunks with an invalid split values
    (like Nan or None) or split values that are not in the list ``split_names``

    Args:
        data: input DataFrame to divide
        split_column: name of the split column. Defaults to "split".
        split_names: list of allowed split names. If the split value is not in it, the
            group is considered unassigned

    Returns:
        tuple with 2 elements
         - list of unassigned DataFrame groups
         - dictionary of assigned DataFrame groups where key is the split name
    """
    split_names = [*split_names]
    if split_column not in data.columns:
        return [df for _, df in data.groupby(level=0, as_index=False, dropna=False)], {}

    unassigned = []
    assigned = {}
    for split_name, df in data.groupby(split_column, dropna=False):
        if split_name in split_names:
            assigned[split_name] = df
        else:
            unassigned.extend([row for _, row in df.groupby(level=0, dropna=False)])
    return unassigned, assigned


def make_atomic_chunks(
    data: pd.DataFrame,
    groups: Iterable[str | pd.Series],
    split_column: str = "split",
    split_names: Iterable[str] = (),
) -> tuple[list[pd.DataFrame], dict[str, pd.DataFrame]]:
    r"""Subdivide the input DataFrame into dissociate chunks from given columns.
    In other words, for two rows in distinct chunks, there will never be the same
    elements in the involved columns, and for two rows in the same chunk, there can be a
    chain of elements all in this chunk to link them. For example, $(A, B)$ and $(C, D)$
    have different values for each column, but if there exist a row $(A, D)$, then we
    can make the chain :math:`(A, B) \rightarrow (A, D) \rightarrow (C, D)`, which
    means the three rows will be in the same chunk.

    Note:
        In the case the data has a ``split`` column with non NaN values, the
        corresponding rows and the chunk they are linked to will be completely assigned
        to that split. However, it will raise an error if a theoretically indivisible
        chunk has rows with different split values.

    Args:
        data: DataFrame to be split into dissociated chunks.
        groups: groups to consider for the dissociation. If group is a string, given
            DataFrame in ``data`` must include a column with this name. If groups is a
            pandas categorical Series, given DataFrame in ``data`` must have the same
            index.
        split_column: Name of the column in ``data`` where the split value will be
            grabbed from. Rows with values within ``split_names`` will be considered
            assigned.
        split_names: Names of wanted splits. rows with split values outside of it will
            be considered unassigned.

    Returns:
        1. List of DataFrames corresponding to the dissociated chunks. concatenating the
           returned DataFrames would end up in the input DataFrame.
        2. dictionary with already assigned atomic chunk, because the "split" value was
           already filled in at least one of the rows
    """
    if not groups:
        return give_already_assigned(data, split_column, split_names)
    else:
        # Reorder groups so that the first group to use groupby method on is the one
        # with the smallest number of unique values
        def group_sort_key(group):
            if isinstance(group, str):
                return len(data[group].unique())
            else:
                return len(group.unique())

        groups = sorted(groups, key=group_sort_key)
        df_list = [
            df
            for _, df in data.groupby(
                groups[0], as_index=False, dropna=False, observed=True
            )
        ]
        if len(groups) > 1:
            for g in groups[1:]:
                unique_values = [
                    set(df[g].unique()) if isinstance(g, str) else set(g.unique())
                    for df in df_list
                ]
                clusters = factorize_sets(unique_values)
                new_df_list = []
                for cluster in clusters:
                    new_df_list.append(pd.concat([df_list[i] for i in cluster]))
                df_list = new_df_list
    unassigned = []
    already_assigned = defaultdict(list)
    if split_column in data.columns:
        for df in df_list:
            split_names = [
                name
                for name in df[split_column].dropna().unique()
                if name in split_names
            ]
            if len(split_names) > 1:
                split_names_str = ", ".join(map(str, split_names))
                warnings.warn(
                    "One chunk has multiple split assignments"
                    f" ({split_names_str}) and will be treated as unassigned",
                    RuntimeWarning,
                )
                unassigned.append(df)
            elif len(split_names) == 1:
                split_name = split_names[0]
                df["split"] = split_name
                already_assigned[split_name].append(df)
            else:
                unassigned.append(df)
    else:
        unassigned = df_list
    already_assigned = {
        name: pd.concat(split) for name, split in already_assigned.items()
    }
    return unassigned, already_assigned
