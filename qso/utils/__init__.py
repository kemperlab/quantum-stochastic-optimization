import pennylane as qml

from jax import numpy as np, Array
from jax.random import PRNGKey, choice
from textwrap import indent


def resample_data(feature_data: Array,
                  response_data: Array,
                  samples: int,
                  key: Array | None = None) -> tuple[Array, Array]:
    """
    Resamples data for bootstrapping.

    Parameters
    ---
    - `feature_data` (`jax.Array`): The feature data to resample. Should be of
      shape `(N, k)`, where `N` is the number of current samples and `k` is the
      number of features.
    - `response_data` (`jax.Array`): The response data to resample. Should be
      of shape `(N,)`, where `N` is the number of current samples.
    - `samples` (`int`): The number of samples to be in the output.
    - `key` (`jax.Array | None`): A random number key required by the `JAX`
      library. If `None` is provided, a seed of `0` is used to create the data.

    Returns
    ---
    - `resampled_feature_data`: The resampled feature 'data with shape
      `(samples, k)`.
    - `resampled_response_data`: The resampled response data with shape
      `(samples,)`.
    """

    n, _ = feature_data.shape
    assert (n, ) == response_data.shape

    key = key if key is not None else PRNGKey(0)
    indices = choice(key, n, shape=(samples, ))

    return feature_data[indices], response_data[indices]


class ProblemHamiltonian:

    def __init__(self, hamiltonian: qml.Hamiltonian) -> None:
        self.hamiltonian = hamiltonian

    def __repr__(self) -> str:
        hamiltonian = str(self.hamiltonian)
        hamiltonian_array = qml.matrix(self.hamiltonian)
        min_eigval = np.linalg.eigvalsh(np.asarray(hamiltonian_array)).min()
        return ("Hamiltonian:\n"
                f"{indent(hamiltonian, ' ' * 4)}\n"
                "Minimum Eigenvalue:\n"
                f"    {min_eigval}")
