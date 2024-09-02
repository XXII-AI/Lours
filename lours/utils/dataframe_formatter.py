from collections.abc import Iterable, Sequence

import pandas as pd
from pandas._typing import Dtype


def reorder_columns(
    input_df: pd.DataFrame, first_columns: Sequence[str], separator: str = "."
) -> pd.DataFrame:
    """Reorder columns of input dataframe with a list of names and a separator character
    indicating what columns are nested

    The first columns will be the one specified in ``first_columns`` and the remaining
    columns will be reordered following alphabetical order, but nested columns (with a
    name that includes a ``separator`` character) are put at the end.

    Args:
        input_df: dataframe which columns are to be reordered
        first_columns: which columns to put first. Will preserve the order of the list,
            even if it's not sorted alphabetically.
        separator: Character used to separate column prefix and value. Used here to
            identify nested columns. Defaults to ".".

    Returns:
        Same dataframe as ``input_df``, except for the order of columns.
    """
    other_columns = set(input_df.columns) - set(first_columns)
    actual_first_columns = [name for name in first_columns if name in input_df.columns]

    def sort_columns(name: str) -> tuple[int, str]:
        return (name.count(separator), name)

    sorted_other_columns = sorted(other_columns, key=sort_columns)
    return input_df[[*actual_first_columns, *sorted_other_columns]]


def set_dataframe_dtypes(
    input_df: pd.DataFrame, dtypes_dict: dict[str, Dtype], nullable_types: Iterable[str]
) -> pd.DataFrame:
    """Set the right dtypes for columns of a DataFrame and a given name->dtype mapping.

    Notes:
        This script will ignore names which corresponding column is not present
        A list of column can be given so that they are deemed nullable. This is
            especially useful for string dtypes, because None is then converted to the
            'None' string.

    Args:
        input_df: input dataframe which columns are to be converted to desired dtype.
        dtypes_dict: dictionary with column names as key and the dtype as value. The
            dtype must be accepted by the :meth:`pandas.DataFrame.astype` method.
        nullable_types: list of columns which are nullable types. As such, for dtypes
            incompatible with null values (like string) the null objects converted to
            strings will be converted back to :obj:`pandas.NA`. Note that if you want
            to use pandas Nullable Integers, you need to explicitly give them in the
            ``dtypes_dict``.
            See https://pandas.pydata.org/docs/user_guide/integer_na.html

    Returns:
        same DataFrame as ``input_df``, except the dtypes of columns have been changed
        to the desired ones.
    """
    dtypes_to_assign = {
        name: dtype for name, dtype in dtypes_dict.items() if name in input_df.columns
    }
    actual_nullable_columns = [t for t in nullable_types if t in dtypes_to_assign]
    null_values = input_df[actual_nullable_columns].isna()
    output_df = input_df.astype(dtypes_to_assign)
    for column_name, column in null_values.items():
        output_df.loc[column, column_name] = pd.NA  # pyright: ignore
    return output_df
