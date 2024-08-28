import pandas as pd
from pandas.testing import assert_frame_equal

from libia.dataset.split import disjoint_groups


def test_cluster_set():
    set_list = [{0, 1}, {2, 3}, {1, 2}, {4, 5}]
    result = [{0, 1, 2}, {3}]
    assert result == disjoint_groups.factorize_sets(set_list)

    set_list = [{0, 1}, {2, 3}, {1, 2}, {4, 5}, {2, 4}]
    result = [{0, 1, 2, 3, 4}]
    assert result == disjoint_groups.factorize_sets(set_list)

    set_list = [{0, 1, 3, 4}, {2}, {4, 5}]
    result = [{0, 2}, {1}]
    assert result == disjoint_groups.factorize_sets(set_list)


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
