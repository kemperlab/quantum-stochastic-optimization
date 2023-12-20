from argparse import Namespace

from jax import Array
from jax.random import KeyArray, PRNGKey, choice


def resample_data(feature_data: Array,
                  response_data: Array,
                  samples: int,
                  key: KeyArray | None = None) -> tuple[Array, Array]:
    n, _ = feature_data.shape
    assert (n, ) == response_data.shape

    key = key if key is not None else PRNGKey(0)
    indices = choice(key, n, shape=(n, ))  # type: ignore

    return feature_data[indices], response_data[indices]
