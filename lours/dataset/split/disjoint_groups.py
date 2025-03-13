import warnings
from collections import defaultdict
from collections.abc import Iterable

import networkx as nx
import pandas as pd


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

    Notes:
        - In the case the data has a ``split`` column with non NaN values, the
          corresponding rows and the chunk they are linked to will be completely
          assigned to that split. However, it will be completely unassigned if a
          theoretically indivisible chunk has rows with different split values.
        - NaN, None or NA values are considered a unique group value, different from
          all other values, NaN or not. This is thus equivalent to having e.g. a UUID.

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
        Tuple with 2 elements:

        - List of DataFrames corresponding to the dissociated chunks.
        - Dictionary with already assigned atomic chunk, because the "split" value was
          already filled in at least one of the rows

        concatenating the returned DataFrames in the list and the dictionary values
        would end up in the input DataFrame.
    """
    if not groups:
        return give_already_assigned(data, split_column, split_names)
    # Reorder groups so that the first group to use groupby method on is the one
    # with the smallest number of unique values
    groups_as_series = [
        data[group] if isinstance(group, str) else group for group in groups
    ]
    chunks_graph = nx.Graph()
    for group in groups_as_series:
        for _, df in data.groupby(group, as_index=False, dropna=True, observed=True):
            nx.add_path(chunks_graph, df.index)
    chunk_indexes = list(nx.connected_components(chunks_graph))
    df_list = [data.loc[list(chunk_index)] for chunk_index in chunk_indexes]
    not_in_chunks = data.loc[list(set(data.index) - set(chunks_graph))]
    # Search for chunks with already assigned split values.
    unassigned, already_assigned = give_already_assigned(
        data=not_in_chunks, split_column=split_column, split_names=split_names
    )
    already_assigned_chunks = defaultdict(list)
    for name, split in already_assigned.items():
        already_assigned_chunks[name].append(split)
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
                df[split_column] = split_name
                already_assigned_chunks[split_name].append(df)
            else:
                unassigned.append(df)
    else:
        unassigned.extend(df_list)
    already_assigned_chunks = {
        name: pd.concat(split) for name, split in already_assigned_chunks.items()
    }
    return unassigned, already_assigned_chunks
