import os
import jax
import pennylane as qml

from jax import numpy as np, Array
from jax.random import PRNGKey, choice
from serde import serde
from dataclasses import dataclass
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


def get_qdev(n_var: int) -> qml.QubitDevice:
    cpu_count = os.cpu_count()
    cpu_count = 1 if cpu_count is None else cpu_count

    if cpu_count <= 4:
        threads = ""
        workers = 1
    elif cpu_count <= 16:
        threads = "4"
        workers = cpu_count // 4
    elif cpu_count <= 64:
        threads = "8"
        workers = cpu_count // 8
    else:
        workers = 8
        threads = str(cpu_count // 8)

    os.environ['OPENBLAS_NUM_THREADS'] = threads
    os.environ['OMP_NUM_THREADS'] = threads
    os.environ['MKL_NUM_THREADS'] = threads

    qdev: qml.QubitDevice = qml.device("default.qubit",
                                       wires=n_var,
                                       max_workers=workers)  # type: ignore

    return qdev


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


@serde
@dataclass
class NormalDistribution:
    mu: float
    sigma: float
    simple: bool = True

    def sample(self, key: Array, shape: tuple[int, ...]) -> Array:
        if self.simple:
            return (self.mu + jax.random.normal(key).item() * np.ones(shape) *
                    self.sigma)
        else:
            return (self.mu + jax.random.normal(key, shape=shape).item() *
                    self.sigma)

    def expected(self, shape: tuple[int, ...]) -> Array:
        return self.mu * np.ones(shape)


Distribution = NormalDistribution
