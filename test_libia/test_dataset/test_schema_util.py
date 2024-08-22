from pathlib import Path

import pandas as pd
import pytest

from libia.dataset.io import schema_util

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_get_enums_default():
    current_schema = schema_util.load_json_schema("default")
    annot_schema = current_schema["properties"]["annotations"]["items"]
    annot_enums = schema_util.get_enums(annot_schema)

    assert annot_enums == {
        "attributes.colors": {
            "red",
            "green",
            "yellow",
            "blue",
            "white",
            "black",
            "orange",
            "purple",
            "grey",
            "brown",
            "pink",
            "beige",
            "cyan",
        },
        "attributes.position": {"front", "back", "side", "top", "unknown"},
    }

    annot_enums_filtered = schema_util.get_enums(annot_schema, ignore_pattern="colors")
    assert annot_enums_filtered == {
        "attributes.position": {"front", "back", "side", "top", "unknown"},
    }


def test_get_enums_nested():
    current_schema = schema_util.load_json_schema("default")
    annot_schema = current_schema["properties"]["annotations"]["items"]
    annot_enums = schema_util.get_enums(annot_schema, separator=">")

    assert annot_enums == {
        "attributes>colors": {
            "red",
            "green",
            "yellow",
            "blue",
            "white",
            "black",
            "orange",
            "purple",
            "grey",
            "brown",
            "pink",
            "beige",
            "cyan",
        },
        "attributes>position": {"front", "back", "side", "top", "unknown"},
    }

    annot_enums_filtered = schema_util.get_enums(
        annot_schema, ignore_pattern="colors", separator=">"
    )
    assert annot_enums_filtered == {
        "attributes>position": {"front", "back", "side", "top", "unknown"},
    }


def test_flatten_schema():
    current_schema = schema_util.load_json_schema("default")
    flattened = schema_util.flatten_schema(current_schema, separator=">")
    # Note : the annotations field is not flattened because it's a list field and thus
    # has no attribute
    assert flattened == [
        "image>file_name",
        "image>id",
        "image>width",
        "image>height",
        "image>tags>time",
        "image>tags>weather",
        "annotations",
    ]


def test_get_dtypes_and_default_values():
    current_schema = schema_util.load_json_schema("default")
    annot_schema = current_schema["properties"]["annotations"]["items"]
    default, dtypes = schema_util.get_dtypes_and_default_values(
        annot_schema, separator=">"
    )
    assert default == {
        "children_ids": [],
        "attributes>colors": [],
        "attributes>position": [],
        "parent_id": pd.NA,
        "tracking_id": pd.NA,
    }

    assert dtypes == {
        "category_id": int,
        "id": int,
        "parent_id": pd.Int64Dtype(),
        "tracking_id": pd.Int64Dtype(),
    }


def test_get_remapping_dict_from_schema():
    current_schema = schema_util.load_json_schema("default")
    annot_schema = current_schema["properties"]["annotations"]["items"]
    remapping_dict = schema_util.get_remapping_dict_from_schema(
        annot_schema, separator=">", prefix="test"
    )
    assert remapping_dict == {
        "id": "test>id",
        "children_ids": "test>children_ids",
        "parent_id": "test>parent_id",
        "tracking_id": "test>tracking_id",
        "category_id": "test>category_id",
        "category_str": "test>category_str",
        "confidence": "test>confidence",
        "bbox": "test>bbox",
        "attributes": {
            "colors": "test>attributes>colors",
            "position": "test>attributes>position",
            "occluded": "test>attributes>occluded",
        },
    }


def test_get_remapping_dict_from_names():
    names = (
        "height",
        "width",
        "relative_path",
        "type",
        "tags.time",
        "tags.weather",
        "split",
    )

    remapping_dict = schema_util.get_remapping_dict_from_names(names)
    assert remapping_dict == {
        "height": "height",
        "width": "width",
        "relative_path": "relative_path",
        "type": "type",
        "tags": {
            "time": "tags.time",
            "weather": "tags.weather",
        },
        "split": "split",
    }

    # There should not be A and A>B fields
    # Also, check that the error is raised whether we call ["A", "A>B"] or ["A>B", "A"]
    with pytest.raises(ValueError):
        schema_util.get_remapping_dict_from_names(("tags", *names))

    with pytest.raises(ValueError):
        schema_util.get_remapping_dict_from_names((*names, "tags"))
