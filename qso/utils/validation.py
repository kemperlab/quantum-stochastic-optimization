from jax import Array
from jax.typing import DTypeLike


def check_ndarray(var_name: str,
                  array: Array,
                  shape: None | tuple[int, ...] = None,
                  dtype: DTypeLike | None = None) -> None:
    """
    Parameters
    ---
    - `var_name` (`str`): The name of the variable in assertation messages.
    - `array` (`jax.Array`): The array to validate.
    - `shape` (`None | tuple[int, ...]`): The shape to assert the array to be.
      If `None`, do not validate shape.
    - `dtype` (`None | jax.typing.DTypeLike`): The `dtype` to assert the array
      to be. If `None`, do not validate `dtype`.
    """

    assert isinstance(array, Array), (
        f"Expected `{var_name}` to be of type `jax.Array`, but found type: {type(array)}"
    )

    if shape is not None:
        assert array.shape == shape, (
            f"Expected `{var_name}` to have shape {shape}, but found shape: {array.shape}"
        )

    if dtype is not None:
        assert array.dtype == dtype, (
            f"Expected `{var_name}` to have `dtype` {dtype}, but found `dtype`: {array.dtype}"
        )
