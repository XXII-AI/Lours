from collections.abc import Iterable
from pathlib import Path
from warnings import warn

from ..dataset import Dataset
from .common import get_image_info, get_images_from_folder, to_dataset_object


def from_folder(
    images_root: str | Path,
    split: str | None = "eval",
    label_map: dict[int, str] | None = None,
    dataset_path: str | Path | None = None,
) -> Dataset:
    """Load a folder of images into a dataset without annotations.

    Globbed image file formats are the following:

     - ".bmp"
     - ".dng"
     - ".jpeg"
     - ".jpg"
     - ".mpo"
     - ".png"
     - ".tif"
     - ".tiff"
     - ".webp"
     - ".pfm"

    Args:
        images_root: Root of folder to get images from
        split: Split value to apply to resulting dataset. Defaults to "eval".
        label_map: Optional label map to apply to dataset. Defaults to None.
        dataset_path: Deprecated name for images_root, if not None, triggers a warning
            and will be removed in future releases

    Returns:
        Dataset with images of given folder, but without annotations.
    """
    if dataset_path is not None:
        warn(
            "Dataset.dataset_path is deprecated in favor of Dataset.images_root",
            DeprecationWarning,
        )
        images_root = dataset_path
    images_root = Path(images_root)
    images_path = get_images_from_folder(images_root)
    if not images_path:
        warn(f"No image file found in folder {images_root}", RuntimeWarning)
    images = []
    for i, img_path in enumerate(images_path):
        image_data = get_image_info(i, img_path, images_root / img_path)
        image_data["split"] = split
        images.append(image_data)
    return to_dataset_object(
        images_root=images_root,
        label_map={} if label_map is None else label_map,
        images=images,
        annotations=[],
    )


def from_files(
    images_root: str | Path = "",
    file_names: str | Path | Iterable[str | Path] = "",
    split: str | None = "eval",
    label_map: dict[int, str] | None = None,
) -> Dataset:
    """Load a list of image paths into a dataset without annotations

    Note:
        Image paths can be globbing patterns as well. As such, if your folder only
        contains .jpg and .png files, calling this function with ``file_names`` set to
        ``["*.jpg", "*.jpeg"]`` will produce the same result as :func:`.from_folder`

    Note:
        Calling this function with ``file_names`` set to "*" is NOT equivalent to
        :func:`.from_folder` because the pattern used in :func:`.from_folder` is more
        constrained.

    Args:
        images_root: Root of folder to get images from. Defaults to "".
        file_names: files to add to the dataset. Can be paths or globbing pattern,
            must be relative to ``images_root``. Defaults to "".
        split: Split value to apply to resulting dataset. Defaults to "eval".
        label_map: Optional label map to apply to dataset. Defaults to None.

    Returns:
        Dataset with given images, but without annotations.
    """
    images_root = Path(images_root)
    if isinstance(file_names, str | Path):
        file_names = [file_names]
    total_image_paths = []
    for f in file_names:
        total_image_paths.extend(images_root.glob(str(f)))
    if not total_image_paths:
        warn(
            f"No image file found in folder {images_root} for the given file paths /"
            " patterns",
            RuntimeWarning,
        )
    images = []
    for i, img_path in enumerate(total_image_paths):
        image_data = get_image_info(i, img_path.relative_to(images_root), img_path)
        image_data["split"] = split
        images.append(image_data)
    return to_dataset_object(
        images_root=images_root,
        label_map={} if label_map is None else label_map,
        images=images,
        annotations=[],
    )
