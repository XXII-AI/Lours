from pathlib import Path

from lours.dataset import Dataset
from lours.utils import parquet_saver


def from_parquet(input_path: Path | str) -> Dataset:
    """Load a Dataset object from a folder with parquet files for its dataframes.

    Other attributes will be loaded from dataset.yaml file in the same folder.

    Args:
        input_path: Folder to read the yaml file and parquet files from.

    Raises:
        ValueError: Raised when the object name contained in
            ``input_dict['__name__']`` is not 'Dataset'.

    Returns:
        Loaded dataset
    """
    data = parquet_saver.dict_from_parquet(Path(input_path))
    if data["__name__"] != "Dataset":
        raise ValueError(
            "Wrong object type for parquet archive. Expected Dataset, got"
            f" {data['__name__']}"
        )
    dataset = Dataset(
        images_root=data["images_root"],
        images=data["images"],
        annotations=data["annotations"],
        label_map=data["label_map"],
    )
    if "booleanized_columns" in data:
        dataset.booleanized_columns = data["booleanized_columns"]

    return dataset
