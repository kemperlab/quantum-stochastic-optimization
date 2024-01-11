import numpy as np
from qso.optimizers.adam import AdamParameters

from qso.problem.feature_selection import FeatureSelectionParameters, FeatureSelectionProblem
from qso.problem.runs import OptimizationRun, ResamplingParameters, Seed
from qso.problem.tight_binding import TightBindingParameters, TightBindingProblem
from qso.utils import NormalDistribution


def test_feature_selection():
    k = 5
    params = FeatureSelectionParameters(
        np.random.randn(k, k).tolist(),
        np.random.randn(k).tolist(), k, k, k)

    run_params = OptimizationRun(
        Seed(1024511202798946),
        2048,
        100,
        ResamplingParameters(0.1, 5, True),
        AdamParameters(),
        params,
    )

    problem = FeatureSelectionProblem(run_params, params)
    hamiltonians_init = problem.get_hamiltonians(10)
    hamiltonians_mid = problem.get_hamiltonians(20)
    hamiltonians_final = problem.get_hamiltonians(20)

    assert hamiltonians_init[:10] == hamiltonians_mid[:10]

    problem.sample_hamiltonian()

    assert hamiltonians_init[:10] == hamiltonians_final[:10]

    problem.sample_hamiltonian()

    assert hamiltonians_mid == hamiltonians_final

    problem.sample_hamiltonian()

    assert hamiltonians_final == problem.common_random_hamiltonians


def test_tight_binding():
    k = 5
    params = TightBindingParameters(
        k,
        {'s'},
        NormalDistribution(10., 1.5),
    )

    run_params = OptimizationRun(
        Seed(1024511202798946),
        2048,
        100,
        ResamplingParameters(0.1, 5, True),
        AdamParameters(),
        params,
    )

    problem = TightBindingProblem(run_params, params)
    hamiltonians_init = problem.get_hamiltonians(10)
    hamiltonians_mid = problem.get_hamiltonians(20)
    hamiltonians_final = problem.get_hamiltonians(20)

    assert hamiltonians_init[:10] == hamiltonians_mid[:10]

    problem.sample_hamiltonian()

    assert hamiltonians_init[:10] == hamiltonians_final[:10]

    problem.sample_hamiltonian()

    assert hamiltonians_mid == hamiltonians_final

    problem.sample_hamiltonian()

    assert hamiltonians_final == problem.common_random_hamiltonians
