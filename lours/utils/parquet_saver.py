from collections.abc import Iterable
from pathlib import Path
from shutil import rmtree

import pandas as pd
from yaml import safe_dump, safe_load

from lours import __version__


def dict_to_parquet(
    output_dict: dict,
    output_dir: Path,
    version: str = __version__,
    fields: Iterable[str] | None = None,
    fields_to_str: Iterable[str] = ("images_root", "relative_path", "images_root"),
    overwrite: bool = False,
) -> None:
    """Save a dictionary containing dataframes as a yaml file and parquet files.

    The dataset can be nested.

    Args:
        output_dict: dictionary to save, containing yaml serializable objects and
            dataframes. Can be nested.
        output_dir: path to folder where to save the yaml and parquets files
        version: data version info for future compatibility.
            Defaults to current Lours version.
        fields: fields to save. Will ignore other fields in the output dictionary.
            If set to None, will save all fields. Defaults to None.
        fields_to_str: fields to convert to str. Useful for non-serializable objects
            like Path
        overwrite: if set to True, will remove the ``output_dir`` directory if it
            already exists. If set to False, will check that the directory either does
            not exist or is empty. Defaults to False

    Raises:
        OSError: Raised when the output directory is not empty and ``overwrite`` is set
            to False
    """
    if overwrite:
        rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)
    elif output_dir.is_dir():
        if any(output_dir.iterdir()):
            raise OSError("Output directory must be empty")
    else:
        output_dir.mkdir(parents=True)

    def replace_dataframes_in_dict(dict_to_convert, prefix):
        converted_dict = {}
        for name, attribute in dict_to_convert.items():
            if fields is not None and name not in fields:
                continue
            if isinstance(attribute, pd.DataFrame):
                parquet_name = f"{prefix}{name}.pq"
                output_parquet_file = output_dir / parquet_name
                converted_dict[name] = f"DataFrame:{parquet_name}"
                attribute.astype(
                    {f: str for f in fields_to_str if f in attribute.columns}
                ).to_parquet(output_parquet_file)
            elif isinstance(attribute, dict):
                converted_dict[name] = replace_dataframes_in_dict(
                    attribute, f"{prefix}{name}."
                )
            else:
                converted_dict[name] = (
                    attribute if name not in fields_to_str else str(attribute)
                )
        return converted_dict

    metadata = replace_dataframes_in_dict(output_dict, "")
    metadata["version"] = version
    with open(output_dir / "metadata.yaml", "w") as f:
        safe_dump(metadata, f)


def dict_from_parquet(
    input_dir: Path,
    fields_to_path: Iterable[str] = ("relative_path", "images_root"),
) -> dict:
    """Create dictionary from folder created with the function :func:`.dict_to_parquet`

    Args:
        input_dir: folder containing yaml and parquet files
        fields_to_path: Iterable of strings to specify which columns will need to be
            converted to Path objects.

    Returns:
        created dictionary. Will be used to reconstruct objects with dataframes, such
        as Dataset or Evaluator.
    """

    def replace_dataframe_placeholders(input_dict):
        for name, attribute in input_dict.items():
            if isinstance(attribute, str) and attribute.startswith("DataFrame:"):
                parquet_path = input_dir / attribute.split(":")[1]
                loaded_dataframe = pd.read_parquet(parquet_path)
                for f in fields_to_path:
                    if f in loaded_dataframe.columns:
                        loaded_dataframe[f] = loaded_dataframe[f].apply(
                            Path  # pyright: ignore
                        )
                input_dict[name] = loaded_dataframe

            elif isinstance(attribute, dict):
                replace_dataframe_placeholders(attribute)

    with open(input_dir / "metadata.yaml") as f:
        metadata = safe_load(f)

    for f in fields_to_path:
        if f in metadata:
            metadata[f] = Path(metadata[f])

    replace_dataframe_placeholders(metadata)
    return metadata
