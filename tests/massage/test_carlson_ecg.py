import numpy as np

from mim.massage.carlson_ecg import (
    extract_field,
    flatten_nested,
    empty_row_count,
    is_ecg_shape_malformed,
    is_beat_shape_malformed,
)


def test_flatten_ridiculous_non_empty_nesting():
    assert flatten_nested([[[[[1], [2]]]]]) == [1, 2]


def test_flatten_ridiculous_empty_nesting():
    assert flatten_nested([[[]]]) == []


def test_ecg_shape():
    good_shapes = [
        (10000, 12),
        (10240, 12),
        (11000, 12),
        (10000, 8),
        (10240, 9)
    ]
    bad_shapes = [
        (9999, 12),
        (10000, 7),
        (0, ),
        (800, 13)
    ]

    for shape in good_shapes:
        assert not is_ecg_shape_malformed(shape)

    for shape in bad_shapes:
        assert is_ecg_shape_malformed(shape)


def test_beat_shape():
    good_shapes = [
        (1200, 12),
        (1300, 12),
        (1250, 8),
        (1200, 9)
    ]
    bad_shapes = [
        (1199, 12),
        (1200, 7),
        (0, ),
        (1200, 12, 2)
    ]

    for shape in good_shapes:
        assert not is_beat_shape_malformed(shape)

    for shape in bad_shapes:
        assert is_beat_shape_malformed(shape)


def test_count_empty_rows():
    assert empty_row_count(np.zeros((100, 10))) == 100
    assert empty_row_count(np.array([[0, 0, 0], [1, 0, 0]])) == 1
    assert empty_row_count(
        np.array(
            [[0, 0.00001, 0, 0, 0],
             [0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0],
             [0.00000000, 0, 0, 0, 0],
             [-1, -1, -1, 0, 0],
             [0, 0, 0, 0, 8]]
        )) == 2


def test_extract_field():
    assert extract_field([1, 2, 3], 'foo') == []
    assert extract_field(np.array([1, 2, 3]), 'foo') == []
    assert extract_field(
        np.array(('a', 'b'), dtype=[('foo', 'O'), ('bar', 'O')]),
        'foo'
    ) == np.array('a')
