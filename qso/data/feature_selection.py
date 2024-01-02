import jax
import jax.numpy as np

from jax import Array
from jax.random import PRNGKey

from ..utils.validation import check_ndarray


def random_linearly_correlated_data(
    samples: int,
    k_real: int,
    k_fake: int,
    k_redundant: int,
    beta: float | Array,
    gamma: float,
    response_vector: np.ndarray,
    redundancy_matrix: np.ndarray,
    key: Array | None = None,
):
    """
    Parameters
    ---
    - `samples` (`int`): The number of samples of generated data to return.
    - `k_real` (`int`): The number of real features to generate.
    - `k_fake` (`int`): The number of fake features to generate.
    - `k_redundant` (`int`): The number of redundant features to generate.
    - `beta` (`float | jax.Array`): The noise associated with each real
      variable, can be a scalar (`float`) is there is a constant noise level or
      a vector (`jax.Array`) with shape `(k_real,)` to specify a noise level
      for each real variable.
    - `gamma` (`float`): The amount of noise in the response variable.
    - `redundancy_matrix` (`jax.Array`): The matrix that specifies the
      correlation betweeen the real variables and the redundant variables. This
      array is expected to have shape `(k_redundant, k_real)`.
    - `response_vector` (`jax.Array`): The vector that specifies the
      correlation betweeen the real variables and the response variable. This
      array is expected to have shape `(k_real,)`.
    - `key` (`jax.Array | None`): A generator for the random noise. If `None`,
      uses a `JAX` key with the seed `0`.

    Returns
    ---
    - data (`jax.Array`): An array of shape `(samples, k_real + k_redundant + k_fake)`.
    - response (`jax.Array`): An array of shape `(samples,)`.

    """

    if key is None:
        key = PRNGKey(0)

    assert samples > 0 and k_real > 0, (
        "Expected `samples` and `k_real` to be ",
        f"positive integers, but found: {samples} and {k_real}, respectively.")

    assert k_fake >= 0 and k_redundant >= 0, (
        "Expected `k_fake` and `k_redundant` to be ",
        f"non-negative integers, but found: {k_fake} and {k_redundant}, respectively."
    )

    if isinstance(beta, Array):
        check_ndarray("beta", beta, shape=(k_real, ))

    check_ndarray("redundancy_matrix",
                  redundancy_matrix,
                  shape=(k_redundant, k_real))
    check_ndarray("response_vector", response_vector, shape=(k_real, ))

    real_vars = jax.random.normal(key, shape=(k_real, samples))
    fake_vars = jax.random.normal(key, shape=(k_fake, samples))

    redundant_vars = redundancy_matrix @ (
        jax.random.normal(key, shape=(k_real, samples)) * beta + real_vars)
    response_vars = response_vector @ (
        jax.random.normal(key, shape=(k_real, samples)) * beta +
        real_vars) + jax.random.normal(key, shape=(samples, )) * gamma

    return np.concatenate([real_vars, fake_vars, redundant_vars],
                          axis=0).T, response_vars
