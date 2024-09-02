"""Set of utility module to manipulate dataset objects."""

import types

from .bbox_converter import column_names_from_format_string

""" Conventions for Lours """
BBOX_COLUMN_NAMES = column_names_from_format_string("XYWH")
DISPLAY_NESTED_COLUMNS = False
DISPLAY_UNBOOLEANIZED = False


def try_import_fiftyone() -> types.ModuleType:
    """Try to import fiftyone and display an informative error if import was
    unsuccessful.

    Raises:
        ImportError: raise an error if the module could not be imported. It can be
            either because it's not installed, or because the database package
            (fyftone-db) is not compatible with the current distribution.

    Returns:
        types.ModuleType: the imported fiftyone module
    """
    try:
        import fiftyone as fo
    except ImportError as e:
        raise ImportError(
            "Fiftyone could not be loaded, make sure you have installed Lours with the"
            " 'fiftyone' extra."
        ) from e
    except Exception as e:
        raise ImportError(
            "Fiftyone is installed but cannot be loaded, you might have to install a"
            " custom 'fyftone-db' for your OS. See"
            " https://docs.voxel51.com/getting_started/troubleshooting.html#alternative-linux-builds"
        ) from e
    return fo
