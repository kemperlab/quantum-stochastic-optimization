from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from .utils.validation import check_ndarray


def random_linearly_correlated_data(
    samples: int,
    k_real: int,
    k_fake: int,
    k_redundant: int,
    beta_i: float | NDArray,
    gamma: float,
    response_vector: np.ndarray,
    redundant_matrix: np.ndarray,
    generator: np.random.Generator | None = None,
):
    """
    Arguments
    ---
    - `samples` (`int`): The number of samples of generated data to return.
    - `k_real` (`int`): The number of real features to generate.
    - `k_fake` (`int`): The number of fake features to generate.
    - `k_redundant` (`int`): The number of redundant features to generate.
    - `beta_i` (`float | numpy.ndarray`): The noise associated with each real
      variable, can be a scalar (`float`) is there is a constant noise level or
      a vector (`numpy.ndarray`) with shape `(k_real,)` to specify a noise
      level for each real variable.
    - `gamma` (`float`): The amount of noise in the response variable.
    - `redundant_matrix` (`numpy.ndarray`): The matrix that specifies
      the correlation betweeen the real variables and the redundant variables.
      This array is expected to have shape `(k_redundant, k_real)`.
    - `response_vector` (`numpy.ndarray`): The vector that specifies the
      correlation betweeen the real variables and the response variable. This
      array is expected to have shape `(k_real,)`.
    - `generator` (`numpy.random.Generator | None`): A generator for the random
      noise. If `None`, instantiates the defaul numpy generator with the seed
      0.

    Returns
    ---
    - data (`numpy.ndarray`): An array of shape `(k_real + k_redundant + k_fake, samples)`.
    - response (`numpy.ndarray`): An array of shape `(samples,)`.

    """

    if generator is None:
        generator = np.random.default_rng(0)

    assert samples > 0 and k_real > 0, (
        "Expected `samples` and `k_real` to be ",
        f"positive integers, but found: {samples} and {k_real}, respectively.")

    assert k_fake >= 0 and k_redundant >= 0, (
        "Expected `k_fake` and `k_redundant` to be ",
        f"non-negative integers, but found: {k_fake} and {k_redundant}, respectively."
    )

    if isinstance(beta_i, np.ndarray):
        check_ndarray("beta_i", beta_i, shape=(k_real, ))

    check_ndarray("redundant_matrix",
                  redundant_matrix,
                  shape=(k_redundant, k_real))
    check_ndarray("response_vector", response_vector, shape=(k_real, ))

    real_vars = generator.standard_normal(size=(k_real, samples))
    fake_vars = generator.standard_normal(size=(k_fake, samples))

    redundant_vars = redundant_matrix @ (
        generator.normal(scale=beta_i, size=(k_real, samples)) + real_vars)
    response_vars = response_vector @ (
        generator.normal(scale=beta_i, size=(k_real, samples)) + real_vars)

    return np.concatenate([real_vars, fake_vars, redundant_vars],
                          axis=0), response_vars

