from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeGuard

import numpy as np
import pandas as pd
from faker import Faker

from ..dataset import Dataset
from .bbox_converter import import_bbox


def construct_attribute_column(
    numpy_generator: np.random.Generator,
    n_rows: int,
    labels: Sequence[str],
    probs: Sequence[float] | None = None,
    is_list_column: bool = True,
) -> pd.Categorical | list[list[str]]:
    """Generate a column with lists of elements taken in a finite pool
    the generated sequence of lists will be in the form of a numpy array, which will
    become a column in a DataFrame.

    Args:
        numpy_generator: numpy random Generator object used to generate random integers
        n_rows: number of rows of generated numpy array
        labels: label strings to use for the attributes
        probs: sequence of probabilities to construct each row. If set to None, will use
            the probabilities by default: for attribute lists, each probability will
            be 0.5, and for simple attribute, probabilities will be evenly distributed.
            Defaults to None.
        is_list_column: if set to True, will construct a column with list of attributes,
            that constitute a subset of the set of labels. Otherwise, will simply
            construct a simple attribute column, where each row is a single label taken
            from ``labels`` according to the probability distribution given by
            ``probs``. Defaults to True.

    Returns:
        list of lists that will be incorporated in a dataframe.
    """
    if is_list_column:
        if probs is None:
            booleanized_column = numpy_generator.integers(
                0, 2, (n_rows, len(labels))
            ).astype(bool)
        else:
            booleanized_column = np.stack(
                list(
                    numpy_generator.choice([True, False], size=n_rows, p=[p_, 1 - p_])
                    for p_ in probs
                ),
                axis=-1,
            )
        labels_np = np.array(list(labels))
        return [list(labels_np[indices]) for indices in booleanized_column]
    else:
        return pd.Categorical(
            numpy_generator.choice(labels, size=n_rows, p=probs), categories=labels
        )


random_attribute_column_type = (
    int
    | Sequence[int]
    | Sequence[str]
    | Sequence[Sequence[float]]
    | dict[str, int]
    | dict[str, Sequence[float]]
    | dict[str, Sequence[str]]
    | dict[str, dict[str, float]]
)
"""The random attribute columns type is a way to design a column with random
attributes.

It will create :math:`N` columns, each :math:`i` th column with :math:`M_i` labels,
the labels being distributed according to the probabilities in :math:`(p_i)_j`
(:math:`(p_i)_j` being of length :math:`M_i`, with values :math:`p_{i,j}` between
0 and 1).

In the case the column is an non-list attribute column, each vector :math:`(p_i)_j` must
addup to 1. Otherwise, each probability :math:`p_{i,j}` is the probability that the
:math:`j` th label of :math:`i` th column is in the attribute list for each cell.

Depending on the type, the values :math:`N`, :math:`M`, :math:`(p_i)_j` and the names
will be constructed differently.

If not specified, column header and labels are randomly generated with
``Faker.unique.word()``

If not specified, the probabilities pi will be either uniform probabilities for
non-list attribute columns, or all set to 0.5 for attribute list columns.

The input can be either

- An integer: :math:`N` is the given integer, :math:`M_i` are random integers between
  2 and 10
- A sequence of integers: :math:`N` is the length of the sequence, :math:`M_i` are the
  integers of that sequence.
- A sequence of str: :math:`N` is the length of the sequence, the column headers are the
  sequence elements, and :math:`M_i` are random integers between 2 and 10.
- A sequence of sequences of float: :math:`N` is the length of the sequence,
  :math:`M_i` is the length of each :math:`i` th sequence, and :math:`(p_i)_j` is the
  :math:`i` th sequence of floats.
- A dictionary of integers: :math:`N` is the length of the dictionary. The column
  headers are the dictionary keys, and :math:`M_i` are the integer values.
- A dictionary of float sequences: :math:`N` is the length of the dictionary. The column
  headers are the dictionary keys, :math:`M_i` is the length of the :math:`i` th float
  sequence, and :math:`(p_i)_j` is the :math:`i` th float sequence
- A dictionary of string sequences: :math:`N` is the length of the dictionary. The
  column headers are the dictionary keys, :math:`M_i` is the length of the :math:`i` th
  string sequence, and the :math:`j` th label of the :math:`i` th column is the
  :math:`j` th element of the :math:`i` th sequence.
- A dictionary of float dictionaries. :math:`N` is the length of the root dictionary.
  The column headers are the dictionary keys, :math:`M_i` is the length of the
  :math:`i` th sub-dictionary,  the :math:`j` th label of the :math:`i` th column is the
  :math:`j` th key of the :math:`i` th sub-dictionary and the probability
  :math:`p_{i,j}` is the corresponding sub-dictionary value
"""


def set_attribute_columns_labels(
    input_dataframe: pd.DataFrame,
    columns_specs: random_attribute_column_type,
    numpy_generator: np.random.Generator,
    fake_generator: Faker,
    is_list: bool = False,
    min_labels: int = 2,
    max_labels: int = 10,
) -> list[str]:
    """From a specification given according to the :obj:`random_attribute_column_type`
    type, add attribute columns to the given dataframe and return the name of added
    columns.

    Depending on ``is_list``, it will be either an attribute column, where each row
    has a single value, taken from a fixed set of possible string labels or an attribute
    list column where each row has a subsset of values from a fixed superset of possible
    string labels.

    Args:
        input_dataframe: DataFrame which will be assigned new columns
        columns_specs: specification of columns, according to the aforementioned
            syntax
        numpy_generator: random generator for numpy arrays
        fake_generator: random generator for random unique words
        is_list: if set to True, will construct list attribute columns. Otherwise, will
            construct simple attribute columns. Defaults to False
        min_labels: When number of labels if not specified, minimum random number of
            labels to generate for the current column. Defaults to 2.
        max_labels: When number of labels if not specified, maximum random number of
            labels to generate for the current column. Defaults to 10.

    Returns:
        The header of added columns. Useful to keep track of list attribute columns
        to booleanize them.
    """

    def random_labels(n_labels: int) -> list[str]:
        return [fake_generator.unique.word() for _ in range(n_labels)]

    def is_float_sequence(
        sequence: Sequence[str] | Sequence[float],
    ) -> TypeGuard[Sequence[float]]:
        types = set(map(type, sequence))
        if types not in [{float}, {str}]:
            raise ValueError(
                "The input specification accepts sequence of only float or only string"
                f" labels for dictionary values, got {types} instead"
            )
        return isinstance(sequence[0], float)

    # This typeguard is needed because typeguard is mostly lacking the type-narrowing
    # feature. See PEP 742 https://peps.python.org/pep-0742/
    def is_str_sequence(
        sequence: Sequence[str] | Sequence[float],
    ) -> TypeGuard[Sequence[str]]:
        return isinstance(sequence[0], str)

    def construct_detailed_column_spec(
        input_spec: int | Sequence[str] | Sequence[float] | dict[str, float],
    ) -> tuple[Sequence[str], Sequence[float] | None]:
        if isinstance(input_spec, int):
            labels = random_labels(n_labels=input_spec)
            probs = None
        elif isinstance(input_spec, dict):
            labels = list(input_spec.keys())
            probs = list(input_spec.values())
        else:
            if is_float_sequence(input_spec):
                labels = random_labels(len(input_spec))
                probs = input_spec
            else:
                assert is_str_sequence(input_spec)
                labels = input_spec
                probs = None
        return labels, probs

    # specification dictionary: key is the name of the column header, and values
    # is two vectors: names and probabilities. probabilities vector can be None
    column_labels: dict[str, tuple[Sequence[str], Sequence[float] | None]] = {}
    if isinstance(columns_specs, int):
        for _ in range(columns_specs):
            n_labels = numpy_generator.integers(min_labels, max_labels)
            header_name = fake_generator.unique.word()
            labels = random_labels(n_labels)
            column_labels[header_name] = (labels, None)
    elif isinstance(columns_specs, dict):
        for header_name, specific_column_spec in columns_specs.items():
            column_labels[header_name] = construct_detailed_column_spec(
                specific_column_spec
            )
    else:  # Simple sequence
        for specific_column_spec in columns_specs:
            if isinstance(specific_column_spec, str):
                header_name = specific_column_spec
                specific_column_spec = numpy_generator.integers(min_labels, max_labels)
            else:
                header_name = fake_generator.unique.word()
            column_labels[header_name] = construct_detailed_column_spec(
                specific_column_spec
            )
    for header_name, (labels, probs) in column_labels.items():
        input_dataframe[header_name] = construct_attribute_column(
            numpy_generator, len(input_dataframe), labels, probs, is_list
        )

    return list(column_labels.keys())


def dummy_dataset(
    n_imgs: int = 2,
    n_annot: int = 2,
    n_labels: int = 3,
    split_names: None | str | Sequence[str] = ("train", "val", "eval"),
    split_shares: Sequence[float] = (0.8, 0.1, 0.1),
    n_list_columns_images: random_attribute_column_type = 0,
    n_list_columns_annotations: random_attribute_column_type = 0,
    n_attribute_columns_images: random_attribute_column_type = 0,
    n_attributes_columns_annotations: random_attribute_column_type = 0,
    booleanize: Literal["all", "random", "none"] = "none",
    keypoints_share: float = 0,
    add_confidence: bool = False,
    generate_real_images: bool = False,
    seed: int = 0,
    **existing_elements,
) -> Dataset:
    """Generate a Dummy dataset for demonstration purpose

    Might also be used for tests

    Args:
        n_imgs: number of frame in the fake dataset
        n_annot: number of annotations
        n_labels: length of the label map
        split_names: sequence containing names of the splits to apply to the dataset as
            a column of images dataframe. If set to None, no "split" column will be
            added to the images dataframe. If empty, will assume all splits are
            ``None``. If not empty, and with 2 elements or more, must be the same size
            as ``split_shares``. Defaults to ``("train", "val", "eval")``.
        split_shares: sequence containing share of each split whose name was given in
            ``split_names``. The ith element in ``split_shares`` represents the share
            (written as a float number between 0 and 1) of the dataset that will be
            assigned to this split. If ``split_names`` is empty or has a length of 1, it
            will be ignored. Otherwise, its size must match length of ``split_names``,
            and the value must all add up to 1. Defaults to ``(0.8, 0.1, 0.1)``.
        n_list_columns_images: Definition of the attribute lists columns for images.
            A list column cell contains a subset of a larger set of possible
            attributes, fixed for the whole columns, in the form of a list or a set.
            These columns are designed to be booleanized and are created with
            the function :func:`~construct_list_column`.
            See :obj:`random_attribute_column_type` for an in depth explanation of the
            syntax. Defaults to 0
        n_list_columns_annotations: number of list columns to add to the annotations
            dataframe. A list column cell contains a subset of a larger set of possible
            attributes, fixed for the whole columns, in the form of a list or a set.
            These columns are designed to be booleanized and are created with
            the function :func:`~construct_list_column`.
            See :obj:`random_attribute_column_type` for an in depth explanation of the
            syntax. Defaults to 0
        n_attribute_columns_images: number of attributes columns to add to the images
            dataframe. An attribute column cell contains one element for a set fixed for
            the whole column. These columns are created with the function
            :func:`~construct_list_column`. See :obj:`random_attribute_column_type`
            for an in depth explanation of the syntax. Defaults to 0
        n_attributes_columns_annotations: number of attributes columns to add to the
            annotations dataframe. An attribute column cell contains one element for a
            set fixed for the whole column. These columns are created with the function
            :func:`~construct_list_column`. See :obj:`random_attribute_column_type`
            for an in depth explanation of the syntax. Defaults to 0
        booleanize: how to booleanize the list columns. Can be "all", "random" and
            "none". Defaults to "none".

            - "all" means all the list columns will converted to multiple boolean
              columns
            - "none" means the list columns will be unchanged
            - "random" means a random number of list columns will be booleanized. The
              number of booleanized columns is chosen randomly, and the choice of
              these n booleanized columns is also done randomly.
        keypoints_share: Share of bounding box which are keypoints, i.e. with a height
            and width of 0. Set it to 1 to only have keypoints, and to 0 to have no
            keypoint. Defaults to 0.
        add_confidence: If set to True, will add a "confidence" column to annotations
            with random values between 0 and 1. Use this option to generate random
            predictions, to be used in e.g. an evaluator. Defaults to False.
        generate_real_images:
            if set to True, will generate random images and save them in the ``/tmp/``
            folder under a random file name. Otherwise, will just generate random
            file path to images without creating any. Defaults to False.
        seed: seed number for the generation. This will ensure that for a given seed
            number, the same dataset will be created.
        **existing_elements: optional existing dataset elements that you want not to be
            random.

    Returns:
        Dummy generated dataset

    Example:
        >>> dummy_dataset()
        Dataset object containing 2 images and 2 objects
        Name :
            inside_else_memory
        Images root :
            such/serious
        Images :
            width  height      relative_path   type  split
        id
        0     342     136       help/me.jpeg  .jpeg  train
        1     377     167  whatever/wait.png   .png  train
        Annotations :
            image_id category_str  category_id  ...  box_y_min   box_width  box_height
        id                                      ...
        0          0         step           15  ...  73.932999   71.552480   42.673983
        1          0          why           19  ...   4.567638  248.551257  122.602211
        <BLANKLINE>
        [2 rows x 8 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}

        Change the seed option to another random dataset following the same rules

        >>> dummy_dataset(seed=1)
        Dataset object containing 2 images and 2 objects
        Name :
            shake_effort_many
        Images root :
            care/suggest
        Images :
            width  height        relative_path  type  split
        id
        0     955     229  determine/story.jpg  .jpg  train
        1     131     840       air/method.bmp  .bmp  train
        Annotations :
            image_id category_str  category_id  ...   box_y_min   box_width  box_height
        id                                      ...
        0          1       listen           14  ...  276.974642    9.718823  184.684056
        1          0        reach           22  ...    6.311037  123.141689  174.239136
        <BLANKLINE>
        [2 rows x 8 columns]
        Label map :
        {14: 'listen', 15: 'marriage', 22: 'reach'}

        Use the ``split_share`` and ``split_names`` to set splits values.
        Use the ``keypoints_share`` option to set a share of bounding box with size of 0

        >>> dataset = dummy_dataset(
        ...     10,
        ...     100,
        ...     split_shares=(0.5, 0.5),
        ...     split_names=("foo", "bar"),
        ...     keypoints_share=0.3,
        ...     add_confidence=True,
        ... )
        >>> dataset
        Dataset object containing 10 images and 100 objects
        Name :
            inside_else_memory
        Images root :
            such/serious
        Images :
            width  height           relative_path   type split
        id
        0     342     645            help/me.jpeg  .jpeg   foo
        1     377     973       whatever/wait.png   .png   foo
        2     136     756        chair/mother.gif   .gif   bar
        3     167     669  someone/challenge.jpeg  .jpeg   foo
        4     114     589  successful/present.bmp   .bmp   bar
        5     257     603           no/where.jpeg  .jpeg   foo
        6     831     941          play/take.tiff  .tiff   foo
        7     684     349           bit/force.gif   .gif   bar
        8     921     834           way/back.tiff  .tiff   bar
        9     553     703      marriage/give.tiff  .tiff   foo
        Annotations :
            image_id category_str  category_id  ...   box_width  box_height  confidence
        id                                      ...
        0          0    interview           25  ...   11.569934  591.860047    0.136767
        1          3         step           15  ...   70.680613  101.235900    0.663684
        2          8    interview           25  ...    0.000000    0.000000    0.749956
        3          5          why           19  ...   99.047865  266.499060    0.163943
        4          0          why           19  ...   69.419403   61.451991    0.689302
        ..       ...          ...          ...  ...         ...         ...         ...
        95         7         step           15  ...  518.765436   55.277118    0.942361
        96         0         step           15  ...    0.000000    0.000000    0.802246
        97         5    interview           25  ...    0.000000    0.000000    0.122368
        98         4          why           19  ...   89.054816  254.947600    0.124429
        99         9          why           19  ...  181.630916   86.810354    0.616242
        <BLANKLINE>
        [100 rows x 9 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}
        >>> (dataset.annotations["box_width"] > 0).value_counts() / dataset.len_annot()
        box_width
        True     0.69
        False    0.31
        Name: count, dtype: float64

        Add list columns, that can be booleanized later

        >>> dummy_dataset(n_list_columns_images=1, n_list_columns_annotations=1)
        Dataset object containing 2 images and 2 objects
        Name :
            inside_else_memory
        Images root :
            such/serious
        Images :
            width  height  ...  split                            discover
        id                 ...
        0     342     136  ...  train                  [chair, challenge]
        1     377     167  ...  train  [someone, beyond, present, enough]
        <BLANKLINE>
        [2 rows x 6 columns]
        Annotations :
            image_id category_str  ...  box_height                                  where
        id                         ...
        0          0         step  ...   42.673983         [take, play, week, force, bit]
        1          0          why  ...  122.602211  [no, season, take, play, choice, bit]
        <BLANKLINE>
        [2 rows x 9 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}

        Or booleanize them right away

        >>> dummy_dataset(
        ...     n_list_columns_images=1, n_list_columns_annotations=1, booleanize="all"
        ... )
        Dataset object containing 2 images and 2 objects
        Name :
            inside_else_memory
        Images root :
            such/serious
        Images :
            width  height  ... discover.present discover.someone
        id                 ...
        0     342     136  ...            False            False
        1     377     167  ...             True             True
        <BLANKLINE>
        [2 rows x 11 columns]
        Annotations :
            image_id category_str  category_id  ... where.season  where.take  where.week
        id                                      ...
        0          0         step           15  ...        False        True        True
        1          0          why           19  ...         True        True       False
        <BLANKLINE>
        [2 rows x 16 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}

        Add attribute columns which then are transformed into categorical columns.

        >>> example = dummy_dataset(
        ...     n_attribute_columns_images={"a": 2, "b": 3},
        ...     n_list_columns_annotations=2,
        ... )
        >>> example
        Dataset object containing 2 images and 2 objects
        Name :
            inside_else_memory
        Images root :
            such/serious
        Images :
            width  height      relative_path   type  split     a      b
        id
        0     342     136       help/me.jpeg  .jpeg  train  play  force
        1     377     167  whatever/wait.png   .png  train  take  force
        Annotations :
            image_id  ...         where
        id            ...
        0          0  ...            []
        1          0  ...  [no, season]
        <BLANKLINE>
        [2 rows x 10 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}
        >>> example.images["b"]
        id
        0    force
        1    force
        Name: b, dtype: category
        Categories (3, object): ['week', 'choice', 'force']

        Instead of integers, use lists of probabilities to steer the distribution of
        attributes.

        >>> example = dummy_dataset(
        ...     200, n_attribute_columns_images=[[0.1, 0.1, 0.8]], seed=1
        ... )
        >>> example
        Dataset object containing 200 images and 2 objects
        Name :
            shake_effort_many
        Images root :
            care/suggest
        Images :
            width  height        relative_path   type  split could
        id
        0      955     488  determine/story.jpg   .jpg  train  note
        1      131     895       air/method.bmp   .bmp  train  firm
        2      229     880   political/lead.jpg   .jpg  train  firm
        3      840     384        like/safe.bmp   .bmp  train  note
        4      953     668      suffer/set.jpeg  .jpeg  train  note
        ..     ...     ...                  ...    ...    ...   ...
        195    122     437    state/almost.tiff  .tiff  train  firm
        196    752     300     weight/tend.jpeg  .jpeg  train  note
        197    554     228  remember/summer.png   .png  train  note
        198    688     605       yet/though.png   .png   eval  note
        199    243     227   describe/road.tiff  .tiff  train  note
        <BLANKLINE>
        [200 rows x 6 columns]
        Annotations :
            image_id category_str  category_id  ...   box_y_min   box_width  box_height
        id                                      ...
        0         77        reach           22  ...   45.427512   40.116677  318.073851
        1        137     marriage           15  ...  202.481384  435.389400  475.375279
        <BLANKLINE>
        [2 rows x 8 columns]
        Label map :
        {14: 'listen', 15: 'marriage', 22: 'reach'}
        >>> example.images["could"].value_counts() / len(example)
        could
        note    0.82
        firm    0.09
        lead    0.09
        Name: count, dtype: float64

        Finally, you can generate fake images as well if you want to test the io
        functions that need images to be valid.

        >>> dataset = dummy_dataset(generate_real_images=True)
        >>> dataset
        Dataset object containing 2 images and 2 objects
        Name :
            inside_else_memory
        Images root :
            /tmp/such/serious
        Images :
            width  height      relative_path   type  split
        id
        0     342     136       help/me.jpeg  .jpeg  train
        1     377     167  whatever/wait.png   .png  train
        Annotations :
            image_id category_str  category_id  ...  box_y_min   box_width  box_height
        id                                      ...
        0          0         step           15  ...  73.932999   71.552480   42.673983
        1          0          why           19  ...   4.567638  248.551257  122.602211
        <BLANKLINE>
        [2 rows x 8 columns]
        Label map :
        {15: 'step', 19: 'why', 25: 'interview'}
        >>> dataset.check()
        Checking Image and annotations Ids ...
        Checking Bounding boxes ..
        Checking label map ...
        Checking images are valid ...
    """
    gen = np.random.default_rng(seed)
    Faker.seed(seed=seed)
    fake_generator = Faker()
    images_root = existing_elements.get(
        "images_root", Path("/".join(fake_generator.words(2)))
    )
    if generate_real_images and not images_root.is_absolute():
        images_root = "/tmp" / images_root
    dataset_name = existing_elements.get(
        "dataset_name", "_".join(fake_generator.words(3))
    )

    if "label_map" in existing_elements:
        label_map = existing_elements["label_map"]
    else:
        label_ids = gen.integers(0, 10 * n_labels, size=n_labels)
        label_map = {
            int(label_id): fake_generator.unique.word() for label_id in label_ids
        }
    if "images" in existing_elements:
        images = existing_elements["images"]
        assert isinstance(images, pd.DataFrame), "Images can only be a dataframe"
        image_ids = images.index
        n_imgs = len(image_ids)
    else:
        image_ids = np.arange(n_imgs)
        image_paths = [
            Path(fake_generator.file_path(depth=1, category="image", absolute=False))
            for i in range(n_imgs)
        ]

        width = gen.integers(100, 1000, size=n_imgs)
        height = gen.integers(100, 1000, size=n_imgs)
        images = pd.DataFrame(
            data={
                "width": width,
                "height": height,
                "relative_path": image_paths,
            },
            index=image_ids,
        )

    annot_ids = np.arange(n_annot)
    annot_image_ids = gen.choice(image_ids, size=n_annot)

    category_ids = gen.choice(a=np.array(list(label_map.keys())), size=n_annot)
    bbox = gen.random((2, 2, n_annot))
    box_x_min, box_y_min = bbox.min(axis=0)
    box_width, box_height = np.abs(bbox[0] - bbox[1])
    if keypoints_share > 0:
        to_zero = gen.choice(
            a=[True, False], size=n_annot, p=[keypoints_share, 1 - keypoints_share]
        )
        box_width[to_zero] = 0
        box_height[to_zero] = 0

    bbox = np.stack([box_x_min, box_y_min, box_width, box_height], axis=1)

    if split_names is not None:
        if isinstance(split_names, str):
            split_names = [split_names]
        n_splits = len(split_names)
        if n_splits == 0:
            images["split"] = None
        elif n_splits == 1:
            images["split"] = split_names[0]
        else:
            if len(split_shares) != n_splits:
                raise ValueError(
                    "Size mismatch between 'split_names' and 'split_shares'"
                    f" ({len(split_names)} vs {len(split_shares)})"
                )
            if sum(split_shares) != 1:
                raise ValueError(
                    "Split share values must addup to 1. Got"
                    f" {sum(split_shares)} instead"
                )
            split = gen.choice(list(split_names), size=n_imgs, p=list(split_shares))
            images["split"] = split
    bbox = import_bbox(
        bbox, images_df=images, image_ids=annot_image_ids, input_format="xywh"
    )
    bbox.index = pd.Index(annot_ids)
    annotations = pd.DataFrame(
        data={
            "image_id": annot_image_ids,
            "category_id": category_ids,
        },
        index=annot_ids,
    )

    if add_confidence:
        annotations["confidence"] = gen.random(n_annot)

    annotations = pd.concat([annotations, bbox], axis="columns")

    columns_to_booleanize = []

    if n_list_columns_images:
        columns_to_booleanize.extend(
            set_attribute_columns_labels(
                input_dataframe=images,
                columns_specs=n_list_columns_images,
                numpy_generator=gen,
                fake_generator=fake_generator,
                is_list=True,
            )
        )

    if n_list_columns_annotations:
        columns_to_booleanize.extend(
            set_attribute_columns_labels(
                input_dataframe=annotations,
                columns_specs=n_list_columns_annotations,
                numpy_generator=gen,
                fake_generator=fake_generator,
                is_list=True,
            )
        )

    if n_attribute_columns_images:
        set_attribute_columns_labels(
            input_dataframe=images,
            columns_specs=n_attribute_columns_images,
            numpy_generator=gen,
            fake_generator=fake_generator,
            is_list=False,
        )

    if n_attributes_columns_annotations:
        set_attribute_columns_labels(
            input_dataframe=annotations,
            columns_specs=n_attributes_columns_annotations,
            numpy_generator=gen,
            fake_generator=fake_generator,
        )

    dataset = Dataset(
        label_map=label_map,
        images=images,
        annotations=annotations,
        images_root=images_root,
        dataset_name=dataset_name,
    )

    if booleanize == "all":
        dataset = dataset.booleanize(columns_to_booleanize)
    elif booleanize == "random":
        subset_to_booleanize = gen.choice(
            columns_to_booleanize,
            size=gen.integers(len(columns_to_booleanize)),
            replace=False,
        )
        dataset = dataset.booleanize(subset_to_booleanize)
    elif booleanize != "none":
        raise ValueError(
            "Invalid booleanize option. Possible values are 'all', 'random' or 'none',"
            f" got {booleanize}"
        )
    if generate_real_images:
        for i, image_data in dataset.images.iterrows():
            image_path = dataset.images_root / image_data["relative_path"]
            format = image_data["type"][1:]
            if format.lower() == "jpg":
                format = "jpeg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            with open(image_path, "wb") as f:
                image_data_stream = fake_generator.image(
                    image_format=format,
                    size=(image_data["width"], image_data["height"]),
                )
                f.write(image_data_stream)
    return dataset
