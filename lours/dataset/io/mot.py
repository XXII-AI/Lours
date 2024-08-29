from pathlib import Path

import imagesize
import pandas as pd

from ..dataset import Dataset


def from_mot(
    ann_txt: Path | str,
    images_folder: Path | str,
    category_id: int,
    category_str: str,
    split: str | None = None,
) -> Dataset:
    """Load a dataset stored in the MOT format.

    See `specifications <https://motchallenge.net/instructions/>`_

    Note:
        The image filenames must represent the image's id (e.g. 0001.jpg). This id
        is used in the .txt annotation file to associate annotations with this image

    Args:
        ann_txt: path to the .txt file containing the MOT annotations.
        images_folder: path to the folder containing the dataset's images which must
            be at the root of this folder.
        category_id: category_id of the objects that are annotated in your MOT dataset.
            this means that your dataset contains only one class, which is the case of
            the MOT datasets we've tested so far (MOT20Det, CroHD).
        category_str: category_str of the objects annotated in your MOT dataset.
        split: split of the loaded dataset.

    Returns:
        Loaded dataset object
    """
    # convert path arguments to Path objects in case they are strings
    ann_txt = Path(ann_txt)
    images_folder = Path(images_folder)

    # load .txt annotations into a dataframe
    ann_df = ann_txt_to_df(ann_txt)
    # add category_id and category_str columns to annotations df
    ann_df["category_id"], ann_df["category_str"] = category_id, category_str
    # drop useless columns from annotations df
    ann_df.drop(columns=["x_world", "y_world"], inplace=True)

    # create images dataframe
    img_df = pd.DataFrame(
        data=[
            [
                int(img_path.stem),  # image id
                Path(img_path.name),  # image name
                img_path,  # image name
                img_path.suffix,  # image type
                split,
            ]
            for img_path in images_folder.glob("*.jpg")
        ],
        columns=["id", "relative_path", "absolute_path", "type", "split"],
    ).set_index("id")

    # add image width and height columns to images df
    img_df["width"] = img_df["absolute_path"].apply(lambda x: imagesize.get(x)[0])
    img_df["height"] = img_df["absolute_path"].apply(lambda x: imagesize.get(x)[1])
    img_df.drop(columns=["absolute_path"], inplace=True)

    # create the dataset's labelmap
    labelmap = {category_id: category_str}

    return Dataset(
        images_root=images_folder,
        annotations=ann_df,
        images=img_df,
        label_map=labelmap,
    )


def ann_txt_to_df(ann_txt: Path | str) -> pd.DataFrame:
    """Read the .txt MOT annotations into a dataframe.

    Args:
        ann_txt: path to .txt annotation of your original MOT dataset
    Returns:
        pd.DataFrame: a dataframe containing the MOT annotations
    """
    # read the .txt annotations into a dataframe.
    ann_df = pd.read_csv(
        ann_txt,
        names=[
            "image_id",
            "obj_id",
            "box_x_min",
            "box_y_min",
            "box_width",
            "box_height",
            "confidence",
            "x_world",
            "y_world",
        ],
        dtype={
            "image_id": int,
            "obj_id": int,
            "box_x_min": float,
            "box_y_min": float,
            "box_width": float,
            "box_height": float,
            "confidence": float,
        },
    )

    # use the dataframe's index as the id of annotations
    ann_df.index.name = "id"

    # Ignore annotations that have a conf=0 as stated in the MOT's webpage:
    # https://motchallenge.net/instructions/
    ann_df = ann_df[ann_df.confidence != 0]

    # drop confidence col if all rows are equal to 1 (dataset has only groundtruths)
    if (ann_df["confidence"] == 1.0).all():
        ann_df.drop(columns=["confidence"], inplace=True)
    # replace confidence values that equals 1 (used for groundtruths) with NaN
    else:
        ann_df["confidence"][ann_df["confidence"] == 1.0] = float("nan")

    return ann_df
