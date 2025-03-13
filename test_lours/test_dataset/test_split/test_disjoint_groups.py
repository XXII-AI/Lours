import pandas as pd
from pandas.testing import assert_frame_equal

from lours.dataset.split import disjoint_groups


def test_atomic_groups():
    cols = ["a", "b", "c"]
    test_df = pd.DataFrame(
        [
            [0, 0, 6],
            [0, 1, 7],
            [1, 2, 8],
            [1, 3, 9],
            [2, 1, 6],
            [2, 2, 7],
            [3, 4, 8],
            [3, 5, 9],
        ],
        columns=cols,
    )
    result_df_list = [
        pd.DataFrame(
            [[0, 0, 6], [0, 1, 7], [1, 2, 8], [1, 3, 9], [2, 1, 6], [2, 2, 7]],
            columns=cols,
        ),
        pd.DataFrame([[3, 4, 8], [3, 5, 9]], columns=cols, index=[6, 7]),
    ]
    to_test, assigned = disjoint_groups.make_atomic_chunks(test_df, ["a", "b"])
    assert not assigned
    assert len(result_df_list) == len(to_test)
    for df1, df2 in zip(result_df_list, to_test):
        assert_frame_equal(df1, df2)


def test_atomic_groups_already_assigned():
    cols = ["a", "b", "c"]
    test_df = pd.DataFrame(
        [
            [0, 0, "a"],
            [0, 1, None],
            [1, 2, None],
            [1, 3, None],
            [2, 1, None],
            [2, 2, None],
            [3, 4, None],
            [3, 5, None],
        ],
        columns=cols,
    )
    result_df_list = [
        pd.DataFrame([[3, 4, None], [3, 5, None]], columns=cols, index=[6, 7]),
    ]
    result_assigned = {
        "a": pd.DataFrame(
            [
                [0, 0, "a"],
                [0, 1, "a"],
                [1, 2, "a"],
                [1, 3, "a"],
                [2, 1, "a"],
                [2, 2, "a"],
            ],
            columns=cols,
        )
    }
    to_test, assigned = disjoint_groups.make_atomic_chunks(
        test_df, ["a", "b"], split_column="c", split_names=["a", "b"]
    )
    assert len(assigned) == len(result_assigned)
    assert_frame_equal(assigned["a"], result_assigned["a"])
    assert len(result_df_list) == len(to_test)
    for df1, df2 in zip(result_df_list, to_test):
        assert_frame_equal(df1, df2)


def test_atomic_groups_three_columns():
    cols = ["a", "b", "c"]
    test_df = pd.DataFrame(
        [
            [0, 0, 6],
            [0, 0, 7],
            [1, 0, 6],
            [1, 0, 7],
            [2, 1, 8],
            [2, 1, 9],
            [3, 1, 8],
            [3, 1, 9],
        ],
        columns=cols,
    )
    result_df_list = [
        pd.DataFrame([[0, 0, 6], [0, 0, 7], [1, 0, 6], [1, 0, 7]], columns=cols),
        pd.DataFrame(
            [[2, 1, 8], [2, 1, 9], [3, 1, 8], [3, 1, 9]],
            columns=cols,
            index=[4, 5, 6, 7],
        ),
    ]
    to_test, assigned = disjoint_groups.make_atomic_chunks(test_df, ["a", "b", "c"])
    assert not assigned
    assert len(result_df_list) == len(to_test)
    for df1, df2 in zip(result_df_list, to_test):
        assert_frame_equal(df1, df2)


def test_atomic_groups_nan_values():
    cols = ["a", "b", "c"]
    test_df = pd.DataFrame(
        [
            [0, None, 6],
            [0, 0, 7],
            [1, 0, 6],
            [1, 0, 7],
            [None, 1, 8],
            [None, None, None],
            [3, 1, 8],
            [3, 1, 9],
            [None, None, None],
        ],
        columns=cols,
    )
    result_df_list = [
        pd.DataFrame([[0, None, 6], [0, 0, 7], [1, 0, 6], [1, 0, 7]], columns=cols),
        pd.DataFrame(
            [[None, 1, 8], [3, 1, 8], [3, 1, 9]],
            columns=cols,
            index=[4, 6, 7],
        ),
        pd.DataFrame([[None, None, None]], columns=cols, index=[5]),
        pd.DataFrame([[None, None, None]], columns=cols, index=[8]),
    ]
    to_test, assigned = disjoint_groups.make_atomic_chunks(test_df, ["a", "b", "c"])
    assert not assigned
    assert len(result_df_list) == len(to_test)
    result_df_list = sorted(result_df_list, key=len)
    to_test = sorted(result_df_list, key=len)
    for df1, df2 in zip(result_df_list, to_test):
        assert_frame_equal(df1, df2)


def test_atomic_groups_nan_values_already_assigned():
    cols = ["a", "b", "c"]
    test_df = pd.DataFrame(
        [
            [None, None, "a"],
            [0, 1, "a"],
            [1, 2, None],
            [1, 3, None],
            [2, 1, None],
            [2, 2, None],
            [3, 4, None],
            [3, 5, None],
        ],
        columns=cols,
    )
    result_df_list = [
        pd.DataFrame([[3.0, 4.0, None], [3.0, 5.0, None]], columns=cols, index=[6, 7]),
    ]
    result_assigned = {
        "a": pd.DataFrame(
            [
                [None, None, "a"],
                [0, 1, "a"],
                [1, 2, "a"],
                [1, 3, "a"],
                [2, 1, "a"],
                [2, 2, "a"],
            ],
            columns=cols,
        )
    }
    to_test, assigned = disjoint_groups.make_atomic_chunks(
        test_df, ["a", "b"], split_column="c", split_names=["a", "b"]
    )
    assert len(assigned) == len(result_assigned)
    assert_frame_equal(assigned["a"], result_assigned["a"])
    assert len(result_df_list) == len(to_test)
    for df1, df2 in zip(result_df_list, to_test):
        assert_frame_equal(df1, df2)
