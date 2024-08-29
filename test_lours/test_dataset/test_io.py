import itertools
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import fiftyone as fo
import pytest
from pandas.testing import assert_frame_equal

from lours.dataset import (
    from_caipy,
    from_caipy_generic,
    from_coco,
    from_coco_keypoints,
    from_crowd_human,
    from_darknet,
    from_darknet_generic,
    from_darknet_yolov5,
    from_files,
    from_folder,
    from_mot,
    from_parquet,
    from_pascalVOC_detection,
    from_pascalVOC_generic,
)
from lours.dataset.io.caipy import load_caipy_annot_folder
from lours.utils.testing import assert_dataset_equal, assert_images_valid

HERE = Path(__file__).parent
DATA = HERE.parent / "test_data"


def test_caipy_io():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    assert dataset.dataset_name == "caipy_dataset"
    dataset.check()

    # Save and reload the dataset,
    # ensure there are no errors raised and that they are the same
    with TemporaryDirectory() as t:
        dataset.to_caipy(t, copy_images=True, to_jpg=True)
        assert_dataset_equal(dataset, from_caipy(dataset_path=t))

    # Save dataset with image links and reload the dataset,
    # ensure images are indeed valid links
    # Finally check that datasets are the same
    with TemporaryDirectory() as t:
        dataset.to_caipy(t, copy_images=False)
        dataset2 = from_caipy(dataset_path=t)
        assert_images_valid(dataset2, assert_is_symlink=True)
        assert_dataset_equal(dataset, dataset2)

    # Save dataset with image links, reload it and re-save it with copy.
    # Ensure images are valid files and not just copied links, even if we remove the
    # first dataset.
    with TemporaryDirectory() as t1:
        with TemporaryDirectory() as t2:
            dataset.to_caipy(t2, copy_images=False)
            from_caipy(dataset_path=t2).to_caipy(t1, copy_images=True)
        # Note that dataset is no longer in folder t2
        dataset3 = from_caipy(dataset_path=t1)
        assert_images_valid(dataset3, assert_is_symlink=False)
        assert_dataset_equal(dataset, dataset3)

    # Save dataset with flattened image names
    with TemporaryDirectory() as t:
        dataset.reset_images_root(DATA).to_caipy(t, flatten_paths=True)
        dataset2 = from_caipy(dataset_path=t)
        dataset2.check()
        assert_frame_equal(dataset2.annotations, dataset2.annotations)
        assert_frame_equal(
            dataset.images.drop("relative_path", axis=1),
            dataset2.images.drop("relative_path", axis=1),
        )
        # Assert no subfolders in Images/{split} nor Annotations/{split}
        subfolders = [
            [
                subfolder
                for subfolder in (Path(t) / data_type / split_name).glob("*")
                if subfolder.is_dir()
            ]
            for (data_type, split_name) in itertools.product(
                ["Images", "Annotations"], ["train", "valid"]
            )
        ]
        assert max(map(len, subfolders)) == 0

    # Save to caipy generic and reload, ensure they are the same
    with TemporaryDirectory() as t_images:
        with TemporaryDirectory() as t_annotations:
            dataset.to_caipy_generic(
                t_images, t_annotations, copy_images=True, to_jpg=True
            )
            dataset2 = from_caipy_generic(t_images, t_annotations)
            assert_dataset_equal(dataset, dataset2)

    # Save to coco and reload, ensure they are the same
    with TemporaryDirectory() as t:
        dataset.to_coco(t, copy_images=True)
        dataset2_train = from_coco(
            coco_json=Path(t) / "annotations_train.json", images_root=t
        )
        dataset2_valid = from_coco(
            coco_json=Path(t) / "annotations_valid.json", images_root=t
        )
        assert_dataset_equal(dataset, dataset2_train + dataset2_valid)

    # Save to darknet and reload, ensure they are the same
    with TemporaryDirectory() as t:
        dataset.to_darknet(t)
        dataset2 = from_darknet(
            t, data_file=Path(t) / "train.data", ids_map=Path(t) / "ids_map.json"
        )
        assert_dataset_equal(dataset, dataset2, ignore_index=True)

    # Save to yolov5 and reload, check they are the same
    with TemporaryDirectory() as t:
        dataset.to_yolov5(t)
        dataset2 = from_darknet(
            t, data_file=Path(t) / "data.yaml", ids_map=Path(t) / "ids_map.json"
        )
        assert_dataset_equal(dataset, dataset2, ignore_index=True)


def test_caipy_io_split_none():
    dataset = from_caipy_generic(
        images_folder=DATA / "caipy_dataset" / "Images" / "train",
        annotations_folder=DATA / "caipy_dataset" / "Annotations" / "train",
        split=None,
    )
    with TemporaryDirectory() as t:
        dataset.to_caipy(t)
        dataset2 = from_caipy(t)

    assert_dataset_equal(dataset, dataset2)


def test_caipy_splits_to_read():
    dataset_train = from_caipy(DATA / "caipy_dataset", splits_to_read="train")
    splits = [split_name for split_name, _ in dataset_train.iter_splits()]
    assert splits == ["train"]
    assert len(dataset_train) == 2
    assert dataset_train.len_annot() == 9

    dataset_valid = from_caipy(DATA / "caipy_dataset", splits_to_read=["valid", "aa"])
    splits = [split_name for split_name, _ in dataset_valid.iter_splits()]
    assert splits == ["valid"]
    assert len(dataset_valid) == 2
    assert dataset_valid.len_annot() == 21

    dataset_full = from_caipy(DATA / "caipy_dataset")
    assert_dataset_equal(dataset_valid + dataset_train, dataset_full)

    dataset_invalid_split = from_caipy(DATA / "caipy_dataset", splits_to_read="aa")
    assert len(dataset_invalid_split) == 0

    dataset_generic_train = from_caipy_generic(
        images_folder=DATA / "caipy_dataset" / "Images",
        annotations_folder=DATA / "caipy_dataset" / "Annotations",
        splits_to_read="train",
    )

    dataset_generic_train2 = from_caipy_generic(
        images_folder=DATA / "caipy_dataset" / "Images" / "train",
        annotations_folder=DATA / "caipy_dataset" / "Annotations" / "train",
        split="train",
    ).reset_images_root(DATA / "caipy_dataset" / "Images")

    assert_dataset_equal(dataset_generic_train, dataset_generic_train2)

    dataset_generic_invalid_split = from_caipy_generic(
        images_folder=DATA / "caipy_dataset" / "Images",
        annotations_folder=DATA / "caipy_dataset" / "Annotations",
        splits_to_read="invalid_split",
    )

    assert len(dataset_generic_invalid_split) == 0


def test_caipy_io_empty_annot():
    dataset = from_folder(DATA / "caipy_dataset" / "Images" / "train", split=None)

    with TemporaryDirectory() as t:
        dataset.to_caipy(t)
        dataset2 = from_caipy(t)

    assert_dataset_equal(dataset, dataset2, remove_na_columns=True)


def test_coco_io():
    val_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )
    train_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_train.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="train",
    )
    dataset = train_dataset + val_dataset
    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    assert dataset.images["type"].unique() == [".png"]
    dataset.check()

    # Save to coco and reload, check they are the same
    with TemporaryDirectory() as t:
        dataset.to_coco(t)
        val_dataset2 = from_coco(
            Path(t) / "annotations_valid.json",
            images_root=DATA / "coco_dataset/data/Images",
            split="valid",
        )
        train_dataset2 = from_coco(
            Path(t) / "annotations_train.json",
            images_root=DATA / "coco_dataset/data/Images",
            split="train",
        )
        assert_dataset_equal(dataset, val_dataset2 + train_dataset2)

    # Save to coco and convert to jpg, check images exist and can be loaded
    # Note that we don't check if they are the same, since png images are converted to
    # jpgs
    with TemporaryDirectory() as t:
        dataset.to_coco(t, copy_images=True, to_jpg=True)
        val_dataset2 = from_coco(
            Path(t) / "annotations_valid.json",
            images_root=Path(t) / "data",
            split="valid",
        )
        train_dataset2 = from_coco(
            Path(t) / "annotations_train.json",
            images_root=Path(t) / "data",
            split="train",
        )
        dataset2 = val_dataset2 + train_dataset2
        assert_images_valid(dataset2)

    # Save to caipy and reload, check they are the same
    with TemporaryDirectory() as t:
        dataset.to_caipy(t, copy_images=True, to_jpg=False)
        dataset2 = from_caipy(t)
        # Caipy label_map is normally smaller because some labels don't get annotations
        # instances
        dataset2.label_map = dataset.label_map
        assert_dataset_equal(dataset, dataset2)

    # Save to darknet and reload, check there are no error
    # Note that we don't check they are the same as darknet has to convert to jpg
    with TemporaryDirectory() as t:
        dataset.to_darknet(t)
        from_darknet(t, data_file=Path(t) / "train.data")


def test_partial_coco_io():
    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_no_labelmap.json",
        images_root=DATA / "coco_dataset/data/Images",
        label_map={9: "object"},
        split="train",
    )
    assert len(dataset) == 2
    assert dataset.len_annot() == 9
    assert dataset.label_map == {9: "object"}
    assert list(dataset.annotations["category_id"].unique()) == [9]
    dataset.check()

    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_empty.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="train",
    )
    assert len(dataset) == 2
    assert dataset.len_annot() == 0
    assert len(dataset.label_map) == 15
    dataset.check()

    dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_empty_no_labelmap.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="train",
    )
    assert len(dataset) == 2
    assert dataset.len_annot() == 0
    assert dataset.label_map == {}
    dataset.check()

    dataset = from_coco_keypoints(
        coco_json=DATA / "coco_dataset/annotations_keypoints.json",
        images_root=DATA / "coco_dataset/data/Images",
        category_name="object",
        split="train",
    )
    assert len(dataset) == 2
    assert dataset.len_annot() == 9
    assert dataset.label_map == {0: "object"}
    assert list(dataset.annotations["category_id"].unique()) == [0]
    dataset.check(allow_keypoints=True)

    with TemporaryDirectory() as t:
        dataset.to_coco(t, copy_images=True, to_jpg=False, box_format="XY")
        with pytest.raises(ValueError):
            from_coco(Path(t) / "annotations.json")

        dataset2 = from_coco(
            Path(t) / "annotations.json",
            images_root=Path(t) / "data",
            split="train",
            box_format="XY",
        )
        assert_images_valid(dataset2)
        assert_dataset_equal(dataset, dataset2)


def test_darknet_io():
    image_info, _ = load_caipy_annot_folder(DATA / "caipy_dataset" / "Annotations")
    assert image_info is not None
    with open(DATA / "darknet_dataset/ids_map.json") as f:
        ids_map = json.load(f)
    dataset = from_darknet(
        dataset_path=DATA / "darknet_dataset",
        data_file=DATA / "darknet_dataset/train.data",
        ids_map=ids_map,
        image_info=image_info,
    )
    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    dataset.check()

    with TemporaryDirectory() as t:
        dataset.to_caipy(t)
        dataset2 = from_caipy(t)
        assert_dataset_equal(dataset, dataset2)

    with TemporaryDirectory() as t:
        dataset.to_coco(t)
        train_dataset2 = from_coco(
            Path(t) / "annotations_train.json", images_root=DATA / "darknet_dataset"
        )
        val_dataset2 = from_coco(
            Path(t) / "annotations_valid.json", images_root=DATA / "darknet_dataset"
        )
        assert_dataset_equal(dataset, train_dataset2 + val_dataset2)

    with TemporaryDirectory() as t:
        dataset.to_darknet(t)
        dataset2 = from_darknet(
            dataset_path=t,
            data_file=Path(t) / "train.data",
            ids_map=Path(t) / "ids_map.json",
            image_info=image_info,
        )
        assert_dataset_equal(dataset, dataset2)


def test_yolov5_io():
    dataset = from_darknet(
        dataset_path=DATA / "yolov5_dataset",
        data_file=DATA / "yolov5_dataset/yolov5.yaml",
    )

    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    dataset.check()

    with TemporaryDirectory() as t:
        dataset.to_yolov5(t)
        dataset2 = from_darknet_yolov5(dataset_path=t)

        assert_dataset_equal(dataset, dataset2)

        dataset3 = from_darknet(dataset_path=t, data_file=Path(t) / "data.yaml")
        assert_dataset_equal(dataset, dataset3)

    with TemporaryDirectory() as t:
        dataset.to_yolov5(t, create_split_folder=True, split_name_mapping={})
        dataset4 = from_darknet_yolov5(dataset_path=t, split_name_mapping={})
        for split_name, split_set in dataset4.iter_splits():
            assert split_name is not None
            assert_dataset_equal(
                dataset.get_split(split_name),
                split_set.reset_images_root(Path(t) / split_name),
            )

    dataset5 = from_darknet_yolov5(
        dataset_path=DATA / "yolov5_dataset",
        data_file=DATA / "yolov5_dataset/yolov5.yaml",
    )

    assert_dataset_equal(dataset, dataset5)

    dataset6 = from_darknet_yolov5(
        data_file=DATA / "yolov5_dataset/yolov5.yaml",
    )

    assert_dataset_equal(dataset, dataset6)

    with pytest.raises(ValueError):
        from_darknet_yolov5(dataset_path=None, data_file=None)

    with pytest.raises(ValueError):
        from_darknet_yolov5()


def test_yolov7_io():
    dataset = from_darknet(
        dataset_path=DATA / "yolov5_dataset",
        data_file=DATA / "yolov5_dataset/yolov5.yaml",
    )

    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    dataset.check()

    with TemporaryDirectory() as t:
        dataset.to_yolov7(t)
        dataset2 = from_darknet(
            dataset_path=Path("/"), data_file=Path(t) / "data.yaml"
        ).reset_images_root(t)

        assert_dataset_equal(dataset, dataset2, ignore_index=True)


def test_darknet_generic():
    names = [
        "person",
        "bear",
        "zebra",
        "bowl",
        "orange",
        "broccoli",
        "chair",
        "potted plant",
        "dining table",
        "tv",
        "microwave",
        "refrigerator",
        "book",
        "clock",
        "vase",
    ]
    dataset = from_darknet_generic(
        images_root=DATA / "yolov5_dataset/train/images",
        labels_root=DATA / "yolov5_dataset/train/labels",
        names=names,
        split="train",
    ) + from_darknet_generic(
        images_root=DATA / "yolov5_dataset/valid/images",
        labels_root=DATA / "yolov5_dataset/valid/labels",
        names=names,
        split="valid",
    )

    assert len(dataset) == 4
    assert dataset.len_annot() == 30
    dataset.check()

    with pytest.raises(FileNotFoundError):
        from_darknet_generic(
            images_root=DATA / "yolov5_dataset/train/images",
            labels_root="fake_folder",
            names=names,
        )


def test_darknet_generic2():
    dataset = from_darknet_generic(
        images_root=DATA / "yolov5_dataset/train/images",
        labels_root=DATA / "yolov5_dataset/train/labels",
        names=["test"],
        split="train",
    )

    assert dataset.label_map[0] == "test"
    for i in range(2, len(dataset.label_map)):
        assert dataset.label_map[i] == str(i)


def test_from_folder():
    dataset = from_folder(images_root=DATA / "yolov5_dataset")

    assert len(dataset) == 4
    assert dataset.len_annot() == 0
    dataset.check()

    fake_label_map = {0: "a", 1: "b"}
    dataset = from_folder(
        images_root=DATA / "yolov5_dataset", label_map=fake_label_map, split=None
    )
    assert len(dataset) == 4
    assert dataset.len_annot() == 0
    assert dataset.label_map == fake_label_map
    dataset.check()


def test_from_empty_folder():
    with pytest.warns(RuntimeWarning):
        dataset = from_folder(images_root=DATA / "caipy_dataset" / "Annotations")

    assert len(dataset) == 0
    assert dataset.len_annot() == 0
    dataset.check()


def test_from_files():
    dataset = from_files(
        images_root=DATA / "yolov5_dataset",
        file_names=["valid/images/*.jpg", Path("train/images/000000000009.jpg")],
    )

    assert len(dataset) == 3
    assert dataset.len_annot() == 0
    dataset.check()

    fake_label_map = {0: "a", 1: "b"}
    dataset = from_files(
        images_root=DATA / "yolov5_dataset",
        file_names=["valid/images/*.jpg", Path("train/images/000000000009.jpg")],
        label_map=fake_label_map,
    )
    assert len(dataset) == 3
    assert dataset.len_annot() == 0
    assert dataset.label_map == fake_label_map
    dataset.check()


def test_to_fiftyone():
    val_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    )

    fo_dataset = val_dataset.to_fiftyone("dataset", "groundtruth")
    assert isinstance(fo_dataset, fo.Dataset)
    assert fo_dataset.name == "dataset"
    assert len(fo_dataset) == len(val_dataset)  # pyright: ignore
    detections = fo_dataset.count_values("groundtruth_detection.detections.label")
    assert isinstance(detections, dict)
    assert val_dataset.len_annot() == sum(detections.values())

    with pytest.raises(FileExistsError):
        val_dataset.to_fiftyone("dataset", "groundtruth2")

    val_dataset.to_fiftyone("dataset", "groundtruth", existing="update")
    detections = fo_dataset.count_values("groundtruth_detection.detections.label")
    assert isinstance(detections, dict)
    assert val_dataset.len_annot() * 2 == sum(detections.values())

    new_fo_dataset = val_dataset.to_fiftyone("dataset", "groundtruth", existing="erase")
    detections = new_fo_dataset.count_values("groundtruth_detection.detections.label")
    assert isinstance(detections, dict)
    assert val_dataset.len_annot() == sum(detections.values())

    fo.delete_dataset("dataset")


def test_to_fiftyone_empty():
    val_dataset = from_coco(
        coco_json=DATA / "coco_dataset/annotations_valid.json",
        images_root=DATA / "coco_dataset/data/Images",
        split="valid",
    ).loc[[]]

    fo_dataset = val_dataset.to_fiftyone("dataset", "groundtruth")
    assert isinstance(fo_dataset, fo.Dataset)
    assert fo_dataset.name == "dataset"
    assert len(fo_dataset) == 0  # pyright: ignore
    assert not fo_dataset.has_field("groundtruth_detection")
    fo.delete_dataset("dataset")


def test_to_fiftyone_debooleanize():
    dataset = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=True,
        json_schema="default",
    )

    fo_dataset = dataset.to_fiftyone("dataset", "groundtruth")
    assert isinstance(fo_dataset, fo.Dataset)
    assert fo_dataset.name == "dataset"
    assert len(fo_dataset) == len(dataset)  # pyright: ignore
    detections = fo_dataset.count_values("groundtruth_detection.detections.label")
    assert isinstance(detections, dict)
    assert dataset.len_annot() == sum(detections.values())
    fo.delete_dataset("dataset")


def test_to_fiftyone_keypoint():
    dataset = from_coco_keypoints(
        coco_json=DATA / "coco_dataset/annotations_keypoints.json",
        images_root=DATA / "coco_dataset/data/Images",
        category_name="object",
        split="train",
    )

    fo_dataset = dataset.to_fiftyone("dataset", "groundtruth", allow_keypoints=True)
    assert isinstance(fo_dataset, fo.Dataset)
    assert fo_dataset.name == "dataset"
    assert len(fo_dataset) == len(dataset)  # pyright: ignore
    keypoints = fo_dataset.count_values("groundtruth_keypoint.keypoints.label")
    assert isinstance(keypoints, dict)
    assert dataset.len_annot() == sum(keypoints.values())
    fo.delete_dataset("dataset")


def test_crowd_human():
    dataset = from_crowd_human(
        annotation_odgt=DATA / "crowdhuman_dataset/crowdhuman_train.odgt"
    )
    assert len(dataset) == 3
    assert dataset.len_annot() == 113
    dataset.check()


def test_pascalvoc():
    dataset = from_pascalVOC_detection(DATA / "pascalvoc_dataset")
    assert len(dataset) == 5
    assert dataset.len_annot() == 18

    dataset_generic = from_pascalVOC_generic(
        annotations_root=DATA / "pascalvoc_dataset" / "Annotations",
        images_root=DATA / "pascalvoc_dataset" / "JPEGImages",
        split_folder=DATA / "pascalvoc_dataset" / "ImageSets" / "Main",
    )
    assert len(dataset_generic) == 6
    assert dataset_generic.len_annot() == 21

    assert set(dataset_generic.images["split"].unique()) == {"train", "val", pd.NA}


def test_mot():
    dataset = from_mot(
        ann_txt=DATA / "mot_dataset/gt.txt",
        images_folder=DATA / "mot_dataset/images",
        category_str="head",
        category_id=0,
    )
    assert len(dataset) == 2
    assert dataset.len_annot() == 80
    dataset.check()


def test_parquet():
    dataset = from_caipy(dataset_path=DATA / "caipy_dataset")
    with TemporaryDirectory() as t:
        dataset.to_parquet(t)
        dataset2 = from_parquet(t)
        with pytest.raises(OSError):
            dataset.to_parquet(t)
        dataset.to_parquet(t, overwrite=True)
    assert_dataset_equal(dataset, dataset2)

    dataset_tags = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=True,
        json_schema="default",
    )
    with TemporaryDirectory() as t:
        dataset_tags.to_parquet(t)
        dataset_tags2 = from_parquet(t)
    assert_dataset_equal(dataset_tags, dataset_tags2)

    dataset_unbooleanized = from_caipy_generic(
        annotations_folder=DATA / "caipy_dataset" / "tags" / "default_schema",
        images_folder=None,
        use_schema=False,
    )
    with TemporaryDirectory() as t:
        dataset_unbooleanized.to_parquet(t)
        dataset_unbooleanized2 = from_parquet(t)

    assert_dataset_equal(dataset_unbooleanized, dataset_unbooleanized2)
