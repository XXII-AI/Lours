"""Registry for known useful preset.

to add a preset, add a new entry in remap_presets
value should be a tuple of two dictionaries,
with a ``old_id -> new_id`` mapping and a ``new_id -> new_name`` mapping
"""

from importlib.resources import files

import pandas as pd

__all__ = ["presets", "list_available_presets"]

presets = {}
presets_folder = files("lours") / "dataset" / "remap_presets"


for p in presets_folder.iterdir():
    if p.is_file() and p.name.endswith(".csv"):
        try:
            input_dataset, output_dataset = p.name.removesuffix(".csv").split("_to_")
        except ValueError as e:
            raise NameError(
                "Badly named csv preset file. Should be in the form "
                f"'<dataset1>_to_<dataset2>.csv', but got {p.name} instead."
            ) from e
        preset_df = pd.read_csv(p.open())
        indexed_preset = preset_df.set_index("input_category_id")
        preset_dict = indexed_preset["output_category_id"].to_dict()
        preset_names = (
            indexed_preset.groupby("output_category_id")["output_category_name"]
            .first()
            .to_dict()
        )
        presets[(input_dataset, output_dataset)] = (preset_dict, preset_names)
        is_invertible = preset_df["output_category_id"].is_unique
        if is_invertible and (output_dataset, input_dataset) not in presets:
            inverted_preset = preset_df.set_index("output_category_id")
            inverted_preset_dict = inverted_preset["input_category_id"].to_dict()
            inverted_preset_names = (
                inverted_preset.groupby("input_category_id")["input_category_name"]
                .first()
                .to_dict()
            )
            presets[(output_dataset, input_dataset)] = (
                inverted_preset_dict,
                inverted_preset_names,
            )


def list_available_presets():
    mapping_names = [f"{in_map}\t->\t{out_map}" for in_map, out_map in presets]
    return "\n".join(mapping_names)
