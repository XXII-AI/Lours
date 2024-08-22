from collections.abc import Iterable

import pandas as pd
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython.core.getipython import get_ipython
from IPython.display import display
from ipywidgets import (
    Dropdown,
    HBox,
    Label,
    Layout,
    Output,
    Stack,
    ToggleButtons,
    VBox,
    jslink,
)
from pandas.io.formats.style_render import CSSDict

from .column_booleanizer import debooleanize

is_notebook = isinstance(get_ipython(), ZMQInteractiveShell)


def convert_columns_to_nested_multi_index(
    columns: "pd.Index[str]", separator: str = "."
) -> pd.MultiIndex:
    """Convert an index with names containing a particular separator
    to a multiIndex. Names without separator will have blank index values for
    level above 0.

    Note:
        This is only for html displaying purpose, the multiIndex is not meant to be
        meaningful

    Args:
        columns: pandas string index of columns names
        separator: separator character to use to split column names into index levels.
            Defaults to ".".

    Returns:
        Multi Index with has many levels as the maximum number of separator character in
        a column name.
    """
    max_nest_level = 1
    nested_names = []
    for name in columns:
        names = name.split(separator)
        max_nest_level = max(max_nest_level, len(names))
        nested_names.append(names)

    def pad_tuple(input_names: list[str]) -> tuple[str, ...]:
        pad_size = max_nest_level - len(input_names)
        return tuple(input_names + [""] * pad_size)

    return pd.MultiIndex.from_tuples([pad_tuple(names) for names in nested_names])


def display_booleanized_dataframe(
    input_df: pd.DataFrame,
    booleanized_columns: Iterable[str] = (),
    separator: str = ".",
) -> None:
    """Utilitary function to display a dataset dataframe properly.

    More specifically, this will try to make easier to read nested data and booleanized
    data.

    In the case data is booleanized or nested, add some interface to let the user chose
    between raw display (the dataframe as it is present in the python code) or formatted
    display, by debooleanizing or transforming the columns into multi index.

    Args:
        input_df: Input dataframe to display. Its column names are potentially very long
            because of booleanized or nested data structure.
        booleanized_columns: List of prefix to search for booleanized columns. The
            ``input_df`` dataframe will be debooleanized for displaying purpose.
            Defaults to ().
        separator: character used to indicate nested data. Will be used for
            debooleanization and column name splitting.
    """
    from . import DISPLAY_NESTED_COLUMNS, DISPLAY_UNBOOLEANIZED

    booleanized_columns = list(booleanized_columns)
    nothing_to_format = not any("." in name for name in input_df.columns)
    if nothing_to_format:
        # No booleanized data, and no nested data. Nothing to format, simply output the
        # dataframe as raw
        display(input_df)
        return
    # Debooleanization can be expensive. Only use the head and tail of dataframe to
    # make this operation cheaper, since it's only for displaying purpose
    if len(input_df) > 100:
        short_df = pd.concat([input_df.iloc[:50], input_df.iloc[:-50]])
    else:
        short_df = input_df.copy()

    # Additional style for nested multiIndex : add dim vertical lines to keep track
    # of the tree structure
    # Note that the type hint is not really necessary, but pyright complains without
    # it. See
    # https://github.com/microsoft/pyright/blob/main/docs/type-concepts.md#generic-types
    nested_table_styles: list[CSSDict] = [
        {"selector": "th", "props": "border-right: 1px solid #F0F0F0"}
    ]

    short_df_nested = short_df.copy()
    short_df_nested.columns = convert_columns_to_nested_multi_index(
        short_df.columns,
        separator=separator,
    )
    short_df_nested = short_df_nested.style.set_table_styles(nested_table_styles)

    short_df_debooleanized = debooleanize(
        short_df, booleanized_columns, separator=separator
    )
    short_df_debooleanized_nested = short_df_debooleanized.copy()
    short_df_debooleanized_nested.columns = convert_columns_to_nested_multi_index(
        short_df_debooleanized.columns, separator=separator
    )
    short_df_debooleanized_nested = (
        short_df_debooleanized_nested.style.set_table_styles(nested_table_styles)
    )

    dataframe_outputs = {
        name: Output() for name in ["raw", "nested", "debool_raw", "debool_nested"]
    }
    with dataframe_outputs["raw"]:
        display(short_df)
    with dataframe_outputs["nested"]:
        display(short_df_nested)
    with dataframe_outputs["debool_raw"]:
        display(short_df_debooleanized)
    with dataframe_outputs["debool_nested"]:
        display(short_df_debooleanized_nested)

    default_dropdown_value = "nested" if DISPLAY_NESTED_COLUMNS else "raw"
    column_format_select = Dropdown(
        options=["raw", "nested"], value=default_dropdown_value
    )

    if booleanized_columns:
        bool_stack = Stack([dataframe_outputs["raw"], dataframe_outputs["nested"]])
        jslink((column_format_select, "index"), (bool_stack, "selected_index"))
        unbool_stack = Stack(
            [
                dataframe_outputs["debool_raw"],
                dataframe_outputs["debool_nested"],
            ]
        )
        jslink(
            (column_format_select, "index"),
            (unbool_stack, "selected_index"),
        )
        # jslink does not work with link between true/false and int index,
        # So we have to get rid of the checkbox to use a toggle buttons instead.
        # See related issue. Given how old it is, we might not have a solution in the
        # near future
        # https://github.com/jupyter-widgets/ipywidgets/issues/1109
        bool_toggle = ToggleButtons(
            options=["yes ", "no "],
            icons=["check", "times"],
            layout=Layout(width="auto"),
            style={"button_width": "auto"},
            value="no " if DISPLAY_UNBOOLEANIZED else "yes ",
        )
        selector = HBox(
            [
                VBox([Label("Booleanize"), Label("Column format")]),
                VBox([bool_toggle, column_format_select]),
            ]
        )
        stack = Stack([bool_stack, unbool_stack])
        jslink((bool_toggle, "index"), (stack, "selected_index"))
    else:
        # No booleanized columns, no point in displaying the booleanized checkbox
        selector = HBox(
            [
                Label("Column format"),
                column_format_select,
            ]
        )
        stack = Stack([dataframe_outputs["raw"], dataframe_outputs["nested"]])
        jslink((column_format_select, "index"), (stack, "selected_index"))

    display(VBox([selector, stack]))
