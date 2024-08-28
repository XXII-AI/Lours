---
orphan: true
---

# Changelog (archived)

```{note}
This CHANGELOG refers to the time this project was maintined internally by XXII.
Since the commit history has been removed for security reasons,
the chaneglog is kept for informational purpose and should not be modified.
The new CHANGELOG is [here](changelog.md)
```

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1 - 2024-06-21]

- Pin numpy version to be <2

## [2.1.0 - 2024-06-21]

### Added

- Add `Dataset.remove_invalid_images` and `Dataset.remove_invalid_annotations` methods.
- Add `mark_origin` and `overwrite_origin` options to `Dataset.merge` method
- Add `from_pascalVOC_detection` and `from_pascalVOC_generic` functions to load pascal datasets
- Add `dataset_regression` fixture for pytest that will test that datasets are the same
- Add more examples to documentation

### Fixed

- Fix spelling errors

### Changed

- Upgrade minimum version to 3.10, so long python 3.9!
- Upgrade pre-commit template and run it
- Change most dataset method return types to `Self` instead of simply `"Dataset"`
- Change classmethod `Dataset.from_template` to be a simple method. Note that this change is not breaking, as `Dataset.from_template(input_dataset, **kwargs)` is equivalent to `input_dataset.from_template(**kwargs)`
- `from_coco` and `from_crowdhuman` both try to parse intelligently the annotation file path to extract both the dataset name and the split name, thanks to a new function `libia.dataset.io.common.parse_annotation_name`
- `Dataset.merge` now automatically convert images root of a dataset to absolute if the other is also absolute
- `to_fiftyone` methods (for dataset and evaluator) now accept a `existing` option to handle existing dataset. You can now erase the existing dataset before uploading yours, or raise an error if it exists. **Possibly breaking** : default behaviour of `to_fiftyone` methods was "update" and is now "error"
- `Dataset.match_index` now accepts a dataset as well as an image dataframe like before
- `Dataset.remap_from_other` now accepts `remove_not_mapped` and `remove_emptied_images` options to remove classes that are not present in the other dataset.
- `Evaluator` now accepts a prediction label map that is neither a subset nor a superset of ground truth label map, and will assume only false negative and false positive for the not mutual classes.
- `dummy_dataset` now accepts options `keypoints_share` and `add_confidence` to make crowd datasets and predictions
- `Dataset.add_annotations` and `annotations_appender.append` now accepts more flexible attributes shapes, and then broadcast them together.

## [2.0.1 - 2024-05-29]

### Added

- Add the possibility to test dataset equality modulo columns that are all NaNs
- Add warning message when label map is incomplete, and complete it with the simple id -> str(id) mapping for missing ids
- Add `check_exhaustive` option to `Dataset.check` and `assert_images_valid` functions

### Fixed

- Fix c2p CLI tool to effectively remove a detection when it is modified
- `Dataset.remove_empty_images` now keeps the dataset name
- add docs for darknet IO
- Suppress some FutureWarning from pandas during tests
- fix bug for caipy when split is `pd.NA` instead of `None` or `np.NaN`
- fix bug when loading caipy with `splits_to_read` set to non existing splits
- Code spelling

## [2.0.0 - 2024-04-02]

### Added

- Add input format option for COCO loading, making it possible to load XY coordinates instead of just bounding boxes
- Add `from_coco_keypoints` function for loading COCO data with points and only one class.
- Add compatibility with caipyjson tags and attributes, and more generally any kind of nested dictionary
- Add column boooleanizer (and debooleanizer) to go from a list objects to columns of boolean value for better queries
- Add Crowd detection evaluator with Mean Average Error metric for count
- Add reindex function
- Add `from_mot` function for loading datasets in MOT format. See <https://motchallenge.net/instructions/>
- Add a method to compute confusion matrix for DetectionEvaluator
- Add reindex function
- Add yolov7 compatibility with a `Dataset.to_yolov7` method.
- Add automatic compliance with schema when saving to caipy
- Add compatibility with caipy splits independently indexed
- Add iterator helper methods to `Dataset` like `Dataset.iter_images` and `Dataset.iter_splits` to make it easier to iterate by a specific attribute
- When loaded with a schema, ``from_caipy`` automatically set missing arrays to the empty list and other fields to their default value specified in the schema when at least one sample in the caipy folder has the field set to a particular value in its caipyjson file, avoiding NaN values in the resulting dataframe.
- Add `to_parquet` and `from_parquet` method to save and load dataset efficiently with pyarrow.
- Add dataframe booleanized columns broadcasting functions, useful for merging datasets
- Add better error messages when calling check functions from `utils.testing`
- Add `remap_from_other` method to remap label map to match another dataset.
- Add `realign_label_map` argument in `Dataset.merge` to avoid incompatible label maps error
- Add `assert_columns_properly_normalized` for caipy json reading
- Add `Dataset.empty()` method to create the same dataset object as before, but with an empty dataframe of annotations. This is useful when creating a prediction dataset.
- Add `AnnotationAppender.reset()` and `AnnotationAppender.finish()` methods to be able to use the annotation appender outside a context window
- Add `category_ids_mapping` optional argument to `AnnotationAppender` and related functions in order to remap the category ids from predictions
- Add `flatten_paths` to cAIpy export function, which lets you save a dataset without subfolders.
- Add `c2f` standalone script to quickly open a caipy dataset into fiftyone
- Add `from_files` function, similar to ``from_folder` but when you already know what files or file patterns you want in the root folder.
- Add `difftools` in `libia.utils` to compute difference between datasets. Useful when we want to update something related to it (like fiftyone)
- Add `libia.utils.doc_utils` for examples in docstring, with a dummy dataset creator
- Add Examples in all methods of `Dataset` object.
- Add `Dataset.reset_index_from_mapping` method to remap index of images and annotationbs dataframes
- **BREAKING** Remove `Dataset.reindex` method and rename it `Dataset.match_index` to avoid confusion with `pandas.reindex`
- Add "See Also" admonitions in many methods to link methods together and to see the related tutorial each time
- Add schemas tutorial

### Changed

- Caipy save is much faster
- Up-to-date dependencies
- `from_coco` function now has `label_map` option in case the categories field is empty in the input json
- `from_coco` assumes `category_id` to be 0 in case it is absent from annotations fields. It will error if it's not absent from ALL annotations though.
- **BREAKING** `Evaluator.predictions` renamed to `Evaluator.predictions_dictionary` for better clarity
- **BREAKING** `DetectionEvaluator.compute_matches` and `DetectionEvaluator.compute_precision_recall` have changed their `predictions` option to `predictions_names` for better clarity.
- `Dataset.merge` now tries to fuse dataframes with overlapping ids, as long as the common subset is the same
- `Dataset.reset_index` now accepts a `start_image_id`.
- **BREAKING** `Dataset.dataset_path` is deprecated in favor `Dataset.images_root`, similar to `Evaluator`.
- Introduce the optional `dataset_name` attribute to be used when dataset name is not the folder name of images root but can be deduced from the loader function, e.g. in `from_caipy`
- dataset merging now merge image indexes before concatenating the annotations. Useful when merging a dataset with annotations and the same dataset with pre-annotations.
- refactor dataset merge logic in a dedicated module
- dataset addition falls back to `realing_label_map` in merge when a `IncompatibleLabelMapsError` is raised.
- add `create_split_folder` option in `dataset_to_darknet` function and related `Dataset` methods, allowing to save all images of a particular split in its dedicated folder.
- `Dataset.get_split` now accept `None` value to get all images with a null split value if needed.
- **BREAKING** `Dataset.remap_from_DataFrame` renamed to `Dataset.remap_from_dataframe`
- Replace warning types from `UserWarning` to the right warning type (`DeprecationWarning` or `RuntimeWarning`)
- Add pandas style `Dataset.loc`, `Dataset.iloc`, `Dataset.loc_annot` and `Dataset.iloc_annot` indexers, along with `filter_images` and `filter_annotations` method.
- Add `record_fo_ids` options in `Dataset.to_fiftyone` and `DetectionEvaluator.to_fiftyone` methods to keep track of fiftyone's UUID of each corresponding image and annotation.
- Add [markdownlint](https://github.com/DavidAnson/markdownlint) [pre-commit hook](https://github.com/igorshubovych/markdownlint-cli#use-with-pre-commit) (and make markdown documents compliant with it)
- Add `--watch` argument in `caipy_to_fiftyone` script to perform live update of fiftyone datasets each time a file is modified in the caipy dataset. Useful when constructing a dataset progressively.
- Add `start_annotations_id` option to `Dataset.reset_index` method.
- Add supplementary checks and formatting to the Dataset basic constructor.
- Add more explanation on crowd counting tutorial.

### Fixed

- Get split does not rely on split being present in annotations anymore
- crowdhuman head visibility is unknown
- Class remapping is now compatible when label map is only a subset of remap dict
- PNG to JPG conversion now works for RGBA images (note that the Alpha channel will be lost)
- `to_yolov5` now automatically convert split values like `eval` and `valid` to their yolov5 accepted equivalent (resp. `test` and `val`)
- fix `DetectionEvalutator.matches` being tied to the class instead of the instance.
- fix dependencies problem: sklearn is in core dependencies and matplotlib in optional "plot-utils" group
- fix yolov7 problem, image path in txt files are also absolute. Please don't use yolov7 export if you don't need to, the dataset specs are terrible.
- diverse pycharm warnings fixed
- type hint of `from_folder` improved
- `from_folder` method does not crash when folder is empty, but returns an empty dataset with a warning.
- Warnings and pyright errors from last pandas version are suppressed
- Use tight layout for confusion matrix plot result
- Use json normalize when loading COCO so that it can be converted to fiftyone
- Skip processing steps when converting an empty dataset to fiftyone or when appending empty annotations to the dataset with the annotation appender context manager
- Prevent annotations index to be reset when using annotations appender
- Prevent loss of dataset name when calling `merge`, `reset_index`, `remap_classes`

### Removed

- libia.model subpackage (dead legacy code) got deleted

## [1.4.0] - 2023-02-01

### Added

- Add CrowdHuman loading module See <https://www.crowdhuman.org/>
- Add `darknet_generic` loading module
- Add more test to improve coverage
- introduce a `BBOX_COLUMN_NAMES` convention for bounding column names in dataset's annotation dataframe

### Fixed

- sum of datasets is now functional and tested (was not working before)

## [1.3.1] - 2023-01-16

### Fixed

- Fix bug regarding confidence subsampling for PR curves
- Proper extremal point for PR curves
- Caipy split stays to None if no split is given when loading and data is in root
- Caipy save keep added attributes during runtime when saving

## [1.3.0] - 2023-01-10

### Added

- Add remove empty images method to dataset
- Add remove emptied images option in remap classes
- Add remove not mapped classes option in remap classes (not mapped were always removed before)
- Add `f_scores_betas` to compute all wanted F-scores, F1, F0.5, F2, etc...

### Changed

- PR curves are now indexed by recall with 101 evenly spaced values between 0 and 1 by default. The old behaviour can be retrieved by setting the option `index_column` to None.
- Reworked evaluation demo
- Improved documentation

## [1.2.0] - 2023-01-06

### Added

- Add bounding box converter
- Add image folder io, when input is simply a folder with images, but no annotation
- load caipy generic does not have to specify an image folder anymore
- conversion to fiftyone for datasets and evaluators
- bugfix regarding annotation index when it's duplicated
- group continuous data with either interval labels (by default), mid-point, mean point or median point

### Changed

- **BREAKING** evaluation predictions and matches are now dictionaries and can be used to evaluate multiple predictions sets at the same time
- **BREAKING** group type alias is now either a column or a ContinuousGroup object (a dictionary that does the same thing but with better checking)

### Fixed

- Fix several failing pyright tests because pandas stubs was updated

## [1.1.0] - 2022-11-17

### Added

- Add caipy generic format
- Add testing module in utils
- More thorough tests for io
- More complete notebook for demo_dataset

### Fixed

- pre-commit's flake8 repo url was moved from gitlab to gitHub

## [1.0.0] - 2022-11-04

### Added

- dataset evaluation tool : see tutorials/demo_evaluation
- dataset split tool : see tutorials/demo_split
- new code checkers, including [pyright](https://github.com/microsoft/pyright) and [pandas stubs](https://github.com/pandas-dev/pandas-stubs)

## [0.2.0] - 2022-07-18

### Added

- Features: Merge, Class remapping, etc.
