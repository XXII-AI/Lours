import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import fiftyone as fo
import pytest
from jsonschema_rs import ValidationError

from lours.dataset import from_caipy_generic

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_caipy_tags():
    with pytest.raises(ValidationError):
        from_caipy_generic(
            annotations_folder=DATA / "caipy_dataset" / "tags" / "custom_schema",
            images_folder=None,
            use_schema=True,
            json_schema="default",
        )
    with pytest.raises(ValidationError):
        from_caipy_generic(
            annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
            images_folder=None,
            use_schema=True,
            json_schema=DATA / "caipy_dataset" / "tags" / "custom_schema.json",
        )

    custom_dataset = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "custom_schema",
        images_folder=None,
        use_schema=True,
        json_schema=DATA / "caipy_dataset" / "tags" / "custom_schema.json",
    )
    dataset = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=True,
        json_schema="default",
    )
    assert len(custom_dataset) == 1
    assert custom_dataset.len_annot() == 2
    assert len(dataset) == 1
    assert dataset.len_annot() == 2
    assert sorted(dataset.images.columns) == sorted(
        [
            "height",
            "width",
            "relative_path",
            "type",
            "tags.time",
            "tags.weather",
        ]
    )

    assert sorted(dataset.annotations.columns) == sorted(
        [
            "image_id",
            "category_id",
            "category_str",
            "box_x_min",
            "box_y_min",
            "box_width",
            "box_height",
            "children_ids",
            "attributes.colors.red",
            "attributes.colors.green",
            "attributes.colors.yellow",
            "attributes.colors.blue",
            "attributes.colors.white",
            "attributes.colors.black",
            "attributes.colors.orange",
            "attributes.colors.purple",
            "attributes.colors.grey",
            "attributes.colors.brown",
            "attributes.colors.pink",
            "attributes.colors.beige",
            "attributes.colors.cyan",
            "attributes.occluded",
            "confidence",
            "parent_id",
            "attributes.position.back",
            "attributes.position.front",
            "attributes.position.side",
            "attributes.position.top",
            "attributes.position.unknown",
        ]
    )
    new_dataset = dataset.debooleanize()
    assert sorted(new_dataset.images.columns) == sorted(dataset.images.columns)
    assert sorted(new_dataset.annotations.columns) == sorted(
        [
            "image_id",
            "category_id",
            "category_str",
            "children_ids",
            "confidence",
            "attributes.position",
            "box_x_min",
            "box_y_min",
            "box_width",
            "box_height",
            "attributes.colors",
            "attributes.occluded",
            "parent_id",
        ]
    )

    with open(DATA / "caipy_dataset" / "tags" / "default_schema" / "785.json") as f:
        initial = json.load(f)

    with TemporaryDirectory() as t:
        dataset.to_caipy_generic(None, t, use_schema=True)
        with open(Path(t) / "785.json") as f:
            result = json.load(f)
    assert result == initial


def test_caipy_tags_to_fiftyone():
    dataset = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=True,
        json_schema="default",
    )
    dataset.to_fiftyone("dataset")
    fo.delete_dataset("dataset")
