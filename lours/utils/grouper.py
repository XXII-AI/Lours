"""Set of functions to construct groups in a dataset, and compute analytics pre group
during e.g. evaluation
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd


@dataclass
class ContinuousGroup:
    """Data Class to encapsulate information to give to the cutting function of pandas
    as parameters, typically used to group continuous data by a limited number of
    groups, similarly to an histogram.

    Depending on the attributes, il will use either :func:`pandas.cut` or
    :func:`pandas.qcut` to give a particular label for each row of you dataframe
    """

    name: str
    """Name of the column to use the cutting function on"""

    bins: float | list[float] = 10
    """value given to the ``bin`` parameter of pandas' function. Can be either a float
    (for the number of bins), or a list of values that will be used as actual bins. Note
    that in the case of :func:`pandas.qcut`, only this attribute being a float makes
    sense."""

    qcut: bool = False
    """Whether to use :func:`pandas.qcut` or :func:`pandas.cut`. Qcut will design the
    bins so that each interval will contain the same number of samples, while cut will
    design the bins so that first and last bins are minimum and maximum value of
    considered column, and all the bins are equally spaced
    (similar to :func:`numpy.linspace`)
    """

    log: bool = False
    """When using cut (and not qcut), whether to separate bins equally in the linear
    space or the log space.
    As such, bins for lower values would be closer to each other"""

    label_type: str = "intervals"
    """What type of label to give to each group given by the cutting function.

    Can be either:
     - "intervals" (default): :class:`pandas.Interval` object usually given as Series
       values by :func:`pandas.cut` and :func:`pandas.qcut`
     - "mid": mid point between the two bins of each interval
     - "mean": mean value of data points comprised in a given interval
     - "median": median value of data points comprised in a given interval

     """

    def to_dict(self) -> dict[str, str | float | list[float] | bool]:
        """Serialize the ContinuousGroup object into a dictionary that can then be used
        as kwargs for :func:`.cut_group`

        Returns:
            Dictionary containing parameters to be read by :func:`.cut_group`
        """
        return {
            "group_name": self.name,
            "bins": self.bins,
            "log": self.log,
            "qcut": self.qcut,
            "label_type": self.label_type,
        }


group = str | ContinuousGroup
"""
Type alias to define a group

Group is either

- the name of a column (for discret groups, such as ``category_id``)
- a :class:`.ContinuousGroup` object to divide continuous data into a given number
  of groups, similar to histograms.

these parameters will be used for the function :func:`lours.util.grouper.cut_group`

Examples:
    Discret group::

        "size"


    Continuous group::

        continuousGroup(name="size", bins=10, log=False, qcut=True)

    Continuous group with bins::

        continuousGroup(name="size", bins=[0, 10, 20, 30], log=False, qcut=False)
"""

group_list = group | Sequence[group]
"""
Group list is either a group or an iterable of groups
"""


def cut_group(
    data: pd.Series | pd.DataFrame,
    group_name: str | None = None,
    bins: int | Iterable[float] = 10,
    label_type: str = "intervals",
    log: bool = False,
    qcut: bool = False,
) -> pd.Series:
    """Cut a dataframe according to one of its column values and criteria
    See :func:`pandas.cut`, :func:`pandas.qcut`

    Args:
        data: Dataframe to extract the column name from
        group_name: name of the column to extract
        bins: parameter used by both :func:`pandas.cut`, :func:`pandas.qcut`. Namely,
            it can be an int to describe the number of bins, or a list of floats, to
            either describe the actual bin edges for :func:`pandas.cut` or the quantile
            edges for :func:`pandas.qcut`
        label_type: what type of label to give to each group given by the cutting
            function.
            Can be either:

            - "intervals" (default): :class:`pandas.Interval` object usually given as
              Series values by :func:`pandas.cut` and :func:`pandas.qcut`
            - "mid": mid-point between the two bins of each interval
            - "mean": mean value of data points comprised in a given interval
            - "median": median value of data points comprised in a given interval

        log: Whether to use logarithmic scale or not, when bins is an integer.
            Useful when the values are not uniformly distributed. Defaults to False.
        qcut: Whether to use :func:`pandas.qcut` instead of :func:`pandas.cut`.
            See corresponding documentation for the differences.
            TL;DR, :func:`pandas.qcut` is based on quantiles (same number of occurrences
            in each bin) while :func:`pandas.cut` is based on values (same interval
            length for each bin). Defaults to False.

    Raises:
        ValueError: Raises an error when log option is selected but the extracted column
            has negative values

    Returns:
        Series with the same length as data, describing a mapping from id to bin. Bin
        labels are Interval Indices describing the upper and lower bound.
        See :class:`pandas.IntervalIndex`
    """
    if isinstance(data, pd.DataFrame):
        assert group_name is not None
        to_cut = data[group_name]
    else:
        to_cut = data
        if group_name is not None:
            to_cut.name = group_name
    cut_function = pd.qcut if qcut else pd.cut
    if (not log) or (not isinstance(bins, int)):
        if isinstance(bins, Iterable):
            bins = [*bins]
        result = cut_function(to_cut, bins)
    else:
        if to_cut.min() < 0:
            raise ValueError("Cannot use log on negative values")
        log_to_cut = np.log(to_cut)
        log_cut = cut_function(log_to_cut, bins)
        assert isinstance(log_cut, pd.Series)

        # Change labels to match actual values and not log ones

        def exp_labels(x: pd.Interval) -> pd.Interval:
            return pd.Interval(np.exp(x.left), np.exp(x.right))

        normal_labels = log_cut.cat.categories.map(exp_labels)
        result = log_cut.cat.rename_categories(normal_labels)
    if label_type == "intervals":
        return result
    elif label_type == "mid":
        return result.apply(lambda x: x.mid)
    elif label_type == "mean":
        means = to_cut.groupby(result, observed=False).mean()
        return result.apply(lambda x: means.loc[x])
    elif label_type == "median":
        means = to_cut.groupby(result, observed=False).median()
        return result.apply(lambda x: means.loc[x])
    raise ValueError("invalid label_type")


@overload
def make_pandas_compatible(
    data: pd.DataFrame, g: str
) -> tuple[str, str, Literal[True]]:
    pass


@overload
def make_pandas_compatible(
    data: pd.DataFrame,
    g: ContinuousGroup,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
) -> tuple[str, pd.Series, Literal[False]]:
    pass


@overload
def make_pandas_compatible(
    data: pd.DataFrame,
    g: group,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
) -> tuple[str, str | pd.Series, bool]:
    pass


def make_pandas_compatible(
    data: pd.DataFrame,
    g: group,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
) -> tuple[str, str | pd.Series, bool]:
    """Construct group from :obj:`group` that will be used for pandas'
    `groupby <https://pandas.pydata.org/docs/reference/groupby.html>`__ method.

    - In the case it's only a name, keep it like that
    - Otherwise, we need to construct an index of data cut according to the given
      bins. This will create a :class:`pandas.Series` with categorical data

    Args:
        data: input DataFrame, must contain the column considered in group ``g``
        g: group depicting a column from ``data`` with potential bins.
            See :obj:`group`
        root_data: Potential root data where some ids in ``data`` refer to a particular.
            columns in ``root_data``. Defaults to None.
        key_to_root: column containing ``root_data`` row ids. Defaults to "image_id".

    Returns:
        Tuple with the 3 following values:

        1. group name
        2. group that can be understood by pandas'
           `groupby <https://pandas.pydata.org/docs/reference/groupby.html>`__ method.
           Can be a simple string referring to a column, or a :class:`pandas.Series`
           with categorical data
        3. boolean indicating whether the group is categorical
           (on which different values are independent of each other)
           or continuous (on which different values represent ranges of a continuous
           value, constructing a discretized histogram)
    """

    def construct_column_group_from_root(
        root_dataframe: pd.DataFrame, input_group_name: str
    ) -> pd.Series:
        column = root_dataframe.loc[data[key_to_root], input_group_name]
        column.index = data.index
        return column

    is_category = True
    if isinstance(g, str):
        group_name = g
        if group_name in data.columns:
            group = g
        else:
            assert root_data is not None
            assert group_name in root_data.columns
            group = construct_column_group_from_root(root_data, g).astype("category")
    else:
        group_name = g.name
        if group_name in data.columns:
            group = cut_group(data, **g.to_dict())  # pyright: ignore
        else:
            assert root_data is not None
            assert group_name in root_data.columns
            group = cut_group(
                construct_column_group_from_root(root_data, group_name),
                **g.to_dict(),  # pyright: ignore
            )
        is_category = False
    return group_name, group, is_category


def get_group_names(groups: group_list) -> list[str]:
    """From a list of groups, get the list of associated names.

    Args:
        groups: single group lor Sequence of groups to extract the names from.

    Returns:
        Names of given groups.
    """
    return [g if isinstance(g, str) else g.name for g in groups_to_list(groups)]


def groups_to_list(groups: group_list) -> list[group]:
    """Convert a single group or Sequence of groups to a list of groups
    (possibly with one element)

    Args:
        groups: Sequence of groups or single groups to convert

    Returns:
        Actual list of groups, more easily handled by other functions.
    """
    if isinstance(groups, str | ContinuousGroup):
        return [groups]
    else:
        return list(groups)


def group_relational_data(
    input_data: pd.DataFrame,
    groups: group_list,
    root_data: pd.DataFrame | None = None,
    key_to_root: str = "image_id",
) -> tuple[dict[str, str | pd.Series], list[str], list[str]]:
    """Create groups that will be applied on ``input_data`` with the
    :meth:`pandas.DataFrame.groupby` method. can be used with a ``root_data`` relational
    DataFrame containing values that we might want to group, provided ``input_data``
    contains a column with reference to a row in ``root_data``.

    Args:
        input_data: DataFrame to group
        groups: groups to apply to ``input_data`` or `root_data``.
            Can be a simple string in the case of categorical data, or a dictionary.
            See :obj:`.group`.
        root_data: DataFrame containing information ``input_data`` may refer to.
            Defaults to None.
        key_to_root: column name in ``input_data`` for the key to ``root_data``.
            Defaults to "image_id".

    Returns:
        3 different objects are returned:

        1. A dictionary with the created groups and their name as a key. The groups can
            be directly used in a `input_data.groupby` call
        2. A list of all category groups, where different values are independent
            from each other
        3. A list of all continuous groups, on which different values represent ranges
            of a continuous value, constructing a discretized histogram

        Note that the two list together should be as long as the group dictionary, and
        their elements must refer to all the actual keys of the dictionary.
    """
    groups = groups_to_list(groups)

    groups_dict: dict[str, str | pd.Series] = {}
    category_groups: list[str] = []
    continuous_groups: list[str] = []
    for g in groups:
        name, group, is_category = make_pandas_compatible(
            input_data, g, root_data, key_to_root
        )
        if is_category:
            category_groups.append(name)
        else:
            assert isinstance(group, pd.Series)
            continuous_groups.append(name)
        groups_dict[name] = group

    return groups_dict, category_groups, continuous_groups
