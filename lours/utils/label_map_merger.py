class IncompatibleLabelMapsError(ValueError):
    """Custom Exception to differentiate ValueError because of incompatible label maps
    from other ValueError exceptions
    """

    pass


def merge_label_maps(
    left: dict[int, str], right: dict[int, str], method: str = "inner"
) -> dict[int, str]:
    """Merge two label maps together, either by trying to include one in the other
    with the inner method, or by constructing the union of the two with the outer method

    Args:
        left: First label map to merge
        right: Second label map to merge
        method: use inner label map or outer

    Raises:
        IncompatibleLabelMapsError: Raised when label maps are incompatible. Namely,
            there exists a ``category_id`` in both label maps, but they have different
            names. In the case of inner method, a IncompatibleLabelMapsError can also be
            raised if the smallest label map is not included in the other.

    Returns:
        Merged label map

    Example:
        >>> label_map1 = {0: "car", 1: "person", 2: "truck"}
        >>> label_map2 = {0: "car", 1: "person"}
        >>> merge_label_maps(label_map1, label_map2)
        {0: 'car', 1: 'person', 2: 'truck'}

        >>> label_map2[3] = "head"
        >>> merge_label_maps(label_map1, label_map2)
        Traceback (most recent call last):
            ...
        lours.utils.label_map_merger.IncompatibleLabelMapsError: Label maps are incompatible

        >>> merge_label_maps(label_map1, label_map2, method="outer")
        {0: 'car', 1: 'person', 2: 'truck', 3: 'head'}

        >>> label_map2[0] = "vehicle"
        >>> merge_label_maps(label_map1, label_map2, method="outer")
        Traceback (most recent call last):
            ...
        lours.utils.label_map_merger.IncompatibleLabelMapsError: Label maps are incompatible
    """
    # Check that the two label maps are compatible.
    # If not you will have to remap one of them
    if len(left) < len(right):
        left, right = right, left
    if method == "inner":
        # If this dataset's object map is the biggest, the other one must be
        # a subset of it (keys and values)
        if not set(right).issubset(set(left)):
            raise IncompatibleLabelMapsError("Label maps are incompatible")
        if right != {k: left[k] for k in right}:
            raise IncompatibleLabelMapsError("Label maps are incompatible")
        return left
    if method == "outer":
        intersection = set(left) & set(right)
        # The other way around when the other dataset's label map is the biggest
        if {k: left[k] for k in intersection} != {k: right[k] for k in intersection}:
            raise IncompatibleLabelMapsError("Label maps are incompatible")
        return left | right
    else:
        raise ValueError("method can only be 'inner' or 'outer'")
