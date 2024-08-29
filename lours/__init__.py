from importlib.metadata import version

__version__ = version("lours")

from . import dataset, evaluation, utils

__all__ = ["dataset", "evaluation", "utils"]
