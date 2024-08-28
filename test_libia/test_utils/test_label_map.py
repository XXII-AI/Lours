from pytest import raises

from libia.utils.label_map_merger import merge_label_maps


def test_merge_label_maps():
    a = {0: "a", 1: "b"}
    b = {1: "b"}
    c = {1: "b", 2: "c"}
    d = {1: "c"}

    merge_label_maps(a, b)
    merge_label_maps(a, c, method="outer")

    with raises(ValueError):
        merge_label_maps(a, c)
        merge_label_maps(a, d)
        merge_label_maps(a, d, method="outer")
