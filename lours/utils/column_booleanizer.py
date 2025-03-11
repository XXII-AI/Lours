from collections.abc import Iterable
from functools import partial

import pandas as pd


def booleanize(
    input_df: pd.DataFrame,
    column_names: str | Iterable[str] | None = None,
    separator: str = ".",
    **possible_values: set | None,
) -> pd.DataFrame:
    """Convert given column in input DataFrame from lists to boolean

    This is mainly used when a particular attribute can have multiple possible
    values at once.

    Every possible value given will be tested to see if it's inside every row's list
    which will give a boolean column.

    In the end, the column will be dropped and N new boolean columns will be created
    with the name in form ``{column_name}{separator}{value}``

    Args:
        input_df: DataFrame on which performing the booleanization. The operation is not
            inplace.
        column_names: columns to convert. After conversion, it will be dropped
            from input DataFrame. Can be either a single string or a list of strings.
        separator: character used to separate original column and value. Defaults to '.'
        **possible_values: kwargs for sets of possible values. Each key in this
            dictionary must match a column name. If the corresponding value is None,
            will deduce it from all occurrences in lists of column given by key.
            Defaults to None.

    Raises:
        KeyError: The given ``column_name`` must be in the columns of ``input_df``
        TypeError: When for a particular column possible values need to be deduced,
            the column must have value that are all iterable except strings.

    Returns:
        New dataset with multiple boolean columns in the form
        ``{column_name}{separator}{value}``.
    """

    def is_true(iterable, value) -> bool:
        try:
            return value in iterable
        except TypeError:
            return value == iterable

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
        return input_df

    for column_name in column_names:
        if input_df[column_name].dropna().apply(lambda x: isinstance(x, str)).any():
            raise TypeError(
                f"Column {column_names} cannot contain a single string, use lists"
                " instead"
            )
        enum = possible_values.get(column_name, None)
        if enum is None:
            enum = set().union(*input_df[column_name].dropna().to_list())
        column_name_index = input_df.columns.get_loc(column_name)
        assert isinstance(column_name_index, int)
        before_columns = input_df.columns[:column_name_index]
        after_columns = input_df.columns[column_name_index + 1 :]
        new_columns = []
        for v in enum:
            booleanized_column_name = f"{column_name}{separator}{v}"
            input_df = input_df.assign(
                **{
                    booleanized_column_name: (
                        input_df[column_name].apply(partial(is_true, value=v))
                    )
                }
            )
            new_columns.append(booleanized_column_name)

        input_df = input_df[
            [
                *before_columns,
                *new_columns,
                *after_columns,
            ]
        ]
    return input_df


def broadcast_booleanization(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    booleanized_columns1: Iterable[str] = (),
    booleanized_columns2: Iterable[str] = (),
    ignore_index: bool = False,
    separator: str = ".",
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Broadcast two dataframes so that they have the same booleanized columns.

    Booleanized columns from ``df1`` that are not present in ``df2`` will be created and
    set to False and vice versa.

    Note: if ``ignore_index`` is set to False, the overlapping ids will be set to the
    value in the other dataframe instead of just False


    Args:
        df1: first dataframe to broadcast
        df2: second dataframe to broadcast
        booleanized_columns1: Columns in ``df1`` that are booleanized. Defaults to ().
        booleanized_columns2: Columns in ``df2`` that are booleanized. Defaults to ().
        ignore_index: if set to True, will create boolean columns full of False
            regardless of index overlap between the two dataframes.
            If set to False, tries to retrieve boolean value in one dataframe
            from the other when creating the column. Defaults to False.
        separator: Character used to separate column prefix and value.
            Defaults to ".".

    Returns:
        tuple containing updated dataframes ``df1`` and ``df2`` with the same
        booleanized columns
    """
    booleanized_columns = set().union(booleanized_columns1, booleanized_columns2)
    for column in booleanized_columns:
        if column not in booleanized_columns2 and column in df2.columns:
            df2 = booleanize(df2, column, separator=separator)
        if column not in booleanized_columns1 and column in df1.columns:
            df1 = booleanize(df1, column, separator=separator)
        bool_columns1 = get_bool_columns(df1, column, separator)
        bool_columns2 = get_bool_columns(df2, column, separator)
        for bool_column in set().union(bool_columns1, bool_columns2):
            if bool_column not in df1.columns:
                df1 = df1.assign(
                    **{
                        bool_column: (
                            False
                            if ignore_index
                            else df2[bool_column].reindex(df1.index, fill_value=False)
                        )
                    }
                )
            if bool_column not in df2.columns:
                df2 = df2.assign(
                    **{
                        bool_column: (
                            False
                            if ignore_index
                            else df1[bool_column].reindex(df2.index, fill_value=False)
                        )
                    }
                )
    return df1, df2, booleanized_columns


def get_bool_columns(
    input_df: pd.DataFrame, column_prefix: str, separator: str = "."
) -> list[str]:
    """Given a prefix and a separator, get all columns that start with
    ``{column_prefix}{separator}``

    This is used in e.g. :func:`.debooleanize`

    Args:
        input_df: DataFrame to get the columns from
        column_prefix: Name of column prefix to retrieve boolean columns.
        separator: Character used to separate column prefix and value.
            Defaults to ".".

    Raises:
        ValueError: Raised when column following the pattern are not boolean

    Returns:
        List of columns that follow the pattern and will be used to construct the list.
    """
    full_prefix = f"{column_prefix}{separator}"

    columns = [
        name
        for name in input_df.columns
        if isinstance(name, str) and name.startswith(full_prefix)
    ]

    column_dtypes = input_df[columns].dtypes

    if any(
        dtype not in [bool, pd.BooleanDtype()]
        for column_name, dtype in column_dtypes.items()
    ):
        raise ValueError(
            f"Expected bool type for columns starting with {column_prefix}, but got"
            f" the following dtypes : {column_dtypes}."
        )
    return columns


def debooleanize(
    input_df: pd.DataFrame,
    column_prefixes: str | Iterable[str],
    separator: str = ".",
) -> pd.DataFrame:
    """Inverse operation of :func:`.booleanize`. Take all columns that start with
    ``{column_prefix}{separator}`` and, assuming they are all boolean columns, convert
    them into a single column of list values.

    Note:
        The column order will be preserved, the debooleanized column will be inserted
        at the same spot the multiple booleanized columns were.

    Args:
        input_df: Input DataFrame we will take the columns from.
        column_prefixes: Name of column prefix (or prefixes) to retrieve boolean
            columns. Also, the name of resulting column (or columns)
        separator: Character used to separate column prefix and value.
            Defaults to ".".

    Raises:
        TypeError: all columns with given prefix must be of boolean dtype

    Returns:
        pd.DataFrame: Resulting DataFrame, with all boolean column which name correspond
            to the prefix drop and a single column added with lists
    """
    if isinstance(column_prefixes, str):
        column_prefixes = [column_prefixes]
    for column_prefix in column_prefixes:
        full_prefix = f"{column_prefix}{separator}"

        columns = get_bool_columns(input_df, column_prefix, separator)
        if not columns:
            continue
        first_column_pos = input_df.columns.get_loc(columns[0])
        columns_df = input_df[columns]
        input_df = input_df.drop(columns=columns)
        columns_df = columns_df.rename(
            columns={n: n.replace(full_prefix, "") for n in columns_df.columns}
        )
        single_column = columns_df.apply(lambda x: x[x].index.tolist(), axis=1)
        input_df = input_df.assign(**{column_prefix: single_column})
        # Now reorder columns so that the newly created column is where the
        # booleanized ones were
        input_df = input_df[
            [
                *input_df.columns[:first_column_pos],
                column_prefix,
                *input_df.columns[first_column_pos:-1],
            ]
        ]
    return input_df
