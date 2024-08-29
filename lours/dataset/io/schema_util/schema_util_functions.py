"""Set of utility function to use json schemas for loading caipy json files

See Also:
    :ref:`Related tutorial </notebooks/6_demo_schemas.ipynb>`
"""

import json
import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import pandas as pd
import requests


def load_json_schema(schema_path: str | Path) -> dict:
    """Load JSON schema file, either from a url or a file path.

    If no schema path or url is given, an example following coco is loaded.

    Args:
        schema_path: Name of internal schema, or path to custom schema.

    Raises:
        KeyError: Errors when a string is given but no corresponding json file is found
            in the ``schemas`` folder.

    Returns:
        Loaded schema dictionary
    """
    if schema_path == "default":
        with (
            files("lours") / "dataset" / "io" / "schema_util" / "default-schema.json"
        ).open() as f:
            return json.load(f)
    if isinstance(schema_path, str) and (
        schema_path.startswith("https://") or schema_path.startswith("http://")
    ):
        response = requests.get(schema_path)
        response.raise_for_status()
        return response.json()
    with open(schema_path) as f:
        return json.load(f)


def get_enums(
    schema: dict, separator: str = ".", ignore_pattern: str = "a^"
) -> dict[str, set]:
    """From a schema, get column names that can be converted to sets of boolean columns.

    Each outputted column will be associated to the list of possible values in output
    dictionary

    Args:
        schema: JSON schema dict describing the expected format of input data
        separator: Separator to apply for path to get flattened paths in the dataset's
            DataFrames. Defaults to ".".
        ignore_pattern: column following this regex pattern will be ignored.
            Defaults to "a^".

    Returns:
        Dictionary describing enum columns and possible values
        (and thus created columns)
    """
    enums = {}
    for name, prop in schema["properties"].items():
        if re.match(ignore_pattern, name) or "type" not in prop:
            continue
        if (
            prop["type"] == "array"
            and "enum" in prop.get("items", {})
            and prop.get("uniqueItems", False)
        ):
            enums[name] = set(prop["items"]["enum"])
        elif prop["type"] == "object":
            sub_enums = get_enums(
                prop, separator=separator, ignore_pattern=ignore_pattern
            )
            for subname, values in sub_enums.items():
                enums[f"{name}{separator}{subname}"] = values
    return enums


def flatten_schema(
    schema: dict, separator: str = ".", prefix: str | None = None
) -> list[str]:
    """From a particular schema, get a list of expected key values if the schema was to
    be flattened by e.g. the function :func:`pandas.json_normalize`

    Note:
        This function is meant to be called recursively, hence the ``prefix`` option.

    Args:
        schema: JSON schema describing expected output format
        separator: Character used to separate name in flattened key. Defaults to ".".
        prefix: Prefix to apply to column names in output dictionary values.
            Defaults to None.

    Returns:
        list of flattened column names.
    """
    keys = []
    for name, prop in schema["properties"].items():
        normalized_name = name if prefix is None else separator.join((prefix, name))
        if prop.get("type") == "object":
            sub_tree = flatten_schema(
                schema=prop, separator=separator, prefix=normalized_name
            )
            keys.extend(sub_tree)
        else:
            keys.append(normalized_name)
    return keys


def get_dtypes_and_default_values(
    schema: dict, separator: str = "."
) -> tuple[dict, dict]:
    """Given a schema, find default values and dtypes to set to a flattened version of
    a dict corresponding to the schema.

    For optional integers and booleans we use pandas' Nullable dtypes when ``np.nan``
    is replaced with ``pd.NA``. Otherwise, these columns will get casted to float as
    soon as a value is missing. See :class:`pandas.BooleanDtype`
    and :class:`pandas.UInt64Dtype`

    Args:
        schema: JSON schema describing expected input format of input dicts.
        separator: Character used to separate name in flattened key. Defaults to ".".

    Returns:
        Dictionary with same keys as the flattened dictionary, and with the default
        values as values. If no default could be found (ambiguous type), the key is not
        present.
    """
    default_values = {}
    dtypes = {}
    dtype_mapping = {
        "integer": {True: "Int64", False: int},
        "bool": {True: "boolean", False: bool},
    }
    flattened_keys = flatten_schema(schema, separator=separator)
    for key in flattened_keys:
        schema_object = schema
        optional = False
        for part in key.split(separator):
            if part not in schema_object.get("required", []):
                optional = True
            schema_object = schema_object["properties"][part]
        default_value = schema_object.get("default")
        key_type = schema_object.get("type", "unknown")
        if optional:
            if default_value is not None:
                default_values[key] = default_value
            elif key_type == "array":
                default_values[key] = []
            elif key_type in ["integer", "bool"]:
                default_values[key] = pd.NA

        if key_type in dtype_mapping:
            dtypes[key] = dtype_mapping[key_type][optional]

    return default_values, dtypes


def fill_with_dtypes_and_default_value(
    schema: dict, input_dataframe: pd.DataFrame, separator: str = "."
) -> pd.DataFrame:
    """Given a schema and dataframe constructed on a list of corresponding dicts,
    avoid having NaN values by setting the default value when possible.

    It is expected that the DataFrame is constructed with :func:`pandas.json_normalize`

    Args:
        schema: JSON schema describing expected input format of input dicts.
        input_dataframe: input dataframe with possible missing values
            (and thus set to NaN)
        separator: Character used to separate name in flattened key. Defaults to ".".

    Returns:
        DataFrame similar to input_dataframe but with NaN replaced with default values
            when possible
    """
    default_values, dtypes = get_dtypes_and_default_values(schema, separator)
    for k, v in default_values.items():
        if k not in input_dataframe.columns:
            continue
        if isinstance(v, list):
            # Note that we don't use fillna here because it does not work with the
            # default value of []
            # See https://stackoverflow.com/questions/33199193/how-to-fill-dataframe-nan-values-with-empty-list-in-pandas
            isnull = input_dataframe[k].isna()
            input_dataframe.loc[isnull, k] = pd.Series([v] * isnull.sum()).values
        else:
            with pd.option_context("future.no_silent_downcasting", True):
                input_dataframe[k] = (
                    input_dataframe[k].fillna(v).infer_objects(copy=False)
                )
    dtypes_to_apply = {
        col: dtype for col, dtype in dtypes.items() if col in input_dataframe.columns
    }
    return input_dataframe.astype(dtypes_to_apply)


def get_remapping_dict_from_schema(
    schema: dict, separator: str = ".", prefix: str | None = None
) -> dict:
    """From a particular schema, get a nested dictionary similar to the expected format
    of given schema.

    Each value of that dictionary will be the name of column to get the value from in
    flattened DataFrame.

    Note:
        This function is meant to b called recursively, hence the ``prefix`` option.

    Args:
        schema: JSON schema describing expected output format
        separator: Character used to separate name in flattened key. Defaults to ".".
        prefix: Prefix to apply to column names in output dictionary values.
            Defaults to None.

    Returns:
        Nested dictionary following format described in schema, and providing mapping
        for nested DataFrames with flattened column names.
    """
    mapping_tree = {}
    for name, prop in schema["properties"].items():
        normalized_name = name if prefix is None else separator.join((prefix, name))
        if prop.get("type") == "object":
            sub_tree = get_remapping_dict_from_schema(
                schema=prop, separator=separator, prefix=normalized_name
            )
            mapping_tree[name] = sub_tree
        else:
            mapping_tree[name] = normalized_name
    return mapping_tree


@lru_cache
def get_remapping_dict_from_names(
    names: frozenset[str] | tuple[str, ...], separator: str = "."
) -> dict[str, list[str]]:
    """From a set of names, get the expected nested dictionary shape, assuming that
    a key with two names separated with the given separator means a nested dictionary
    shape.

    For example "a.b" means output shape is of the form ``{a: {b: value}}``

    Note:
        For the LRU cache to be used, the given names must hashable, either tuple or
        frozenset.

    Args:
        names: Set of names to parse the underlying structure from.
        separator: Character used to separate name in flattened key. Defaults to ".".

    Returns:
        Nested remapping dictionary with values set to flattened dictionary key to
        take values from.
    """
    output = {}
    for name in names:
        keys = name.split(separator)
        current = output
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                if k in current:
                    raise ValueError(
                        f"Cannot assign two values to key {name}."
                        f" Remapping dict state : {output}"
                    )
                current[k] = name
            else:
                if k not in current.keys():
                    current[k] = {}
                elif not isinstance(current[k], dict):
                    raise ValueError(
                        f"Cannot assign both a value and a dict to key {name}."
                        f" Remapping dict state : {output}"
                    )
                current = current[k]
    return output


def remap_dict(flattened_dict: dict, mapping_tree: dict | None = None) -> dict:
    """From a mapping tree, convert a flattened dict, possibly taken from a DataFrame
    into a nested dictionary.

    Args:
        flattened_dict: dictionary without sub-dictionary, easily readable by pandas.
        mapping_tree: nested dictionary following expected output shape.
            Each value represents. the key name from flattened dictionary to take the
            value from. If set to None, will deduce it from the key names and separator
            character ".". Defaults to None.

    Returns:
        Remapped nested dictionary
    """
    output_dict = {}
    if mapping_tree is None:
        mapping_tree = get_remapping_dict_from_names(frozenset(flattened_dict.keys()))
    for k, v in mapping_tree.items():
        if isinstance(v, dict):
            output_dict[k] = remap_dict(flattened_dict, v)
        else:
            output_value = flattened_dict.get(v, None)
            # Remove both empty lists and NaN/None values
            if isinstance(output_value, list):
                if not output_value:
                    continue
            else:
                isna = pd.isna(output_value)
                try:
                    if isna:
                        continue
                except ValueError:
                    # pd.isna either outputs a bool or a bool array when the input is
                    # iterable. In that case this raises a ValueError
                    # (Ambiguous truth value), which we ignore because then the object
                    # is clearly not na
                    if isna.any():  # pyright: ignore
                        raise ValueError(f"value contains nan : {output_value}")
                    pass
            output_dict[k] = output_value
    return output_dict
