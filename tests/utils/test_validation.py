import numpy as np

from qso.utils.validation import check_ndarray


def test_check_ndarray():
    array = np.array([[0.], [1.], [205.]], dtype=np.float32)

    check_ndarray("array", array, shape=(3, 1), dtype=np.float32)


def test_check_ndarray_incorrect_shape():
    array = np.array([[0.], [1.], [205.]], dtype=np.float32)

    try:
        check_ndarray("array", array, shape=(3, 2), dtype=np.float32)

    except AssertionError as e:
        assert e.args == (
            "Expected `array` to have shape (3, 2), but found shape: (3, 1)", )


def test_check_ndarray_incorrect_dtype():
    array = np.array([[0.], [1.], [205.]], dtype=np.float32)

    try:
        check_ndarray("array", array, shape=(3, 1), dtype=np.float64)

    except AssertionError as e:
        assert e.args == (
            "Expected `array` to have `dtype` <class 'numpy.float64'>, but found `dtype`: float32",
        )


def test_check_ndarray_not_ndarray():
    array = np.array([[0.], [1.], [205.]], dtype=np.float32)

    try:
        check_ndarray(
            "array",
            array,  # type: ignore
            shape=(3, 1),
            dtype=np.float32)

    except AssertionError as e:
        assert e.args == (
            "Expected `array` to be of type `numpy.ndarray`, but found type: <class 'list'>",
        )
