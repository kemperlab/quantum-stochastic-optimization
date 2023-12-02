import numpy as np
from numpy.typing import NDArray, DTypeLike


def check_ndarray(var_name: str,
                  array: NDArray,
                  shape: None | tuple[int, ...] = None,
                  dtype: DTypeLike | None = None) -> None:
    """
    Arguments
    ---
    - `var_name` (`str`): The name of the variable in assertation messages
    - `array` (`numpy.ndarray`): The array to validate
    - `shape` (`None | tuple[int, ...]`): The shape to assert the array to be.
      If `None`, do not validate shape.
    - `dtype` (`None | numpy.typing.DTypeLike`): The `dtype` to assert the array to be. If `None`, do not validate `dtype`.
    """

    assert isinstance(array, np.ndarray), (
        f"Expected `{var_name}` to be of type `numpy.ndarray`, but found type: {type(array)}"
    )

    if shape is not None:
        assert array.shape == shape, (
            f"Expected `{var_name}` to have shape {shape}, but found shape: {array.shape}"
        )

    if dtype is not None:
        assert array.dtype == dtype, (
            f"Expected `{var_name}` to have `dtype` {dtype}, but found `dtype`: {array.dtype}"
        )
