from random import randbytes
from serde import serde, InternalTagging
from dataclasses import dataclass
from qso.problem.feature_selection import FeatureSelectionParameters, FeatureSelectionProblem

from qso.problem.problem import QSOProblem
from qso.problem.tight_binding import TightBindingParameters, TightBindingProblem

from .optimizers import OptimizerParameters
from .problem import ProblemParameters


@serde
class Random:
    pass


@serde
class RunNumber:
    pass


@serde
@dataclass
class Seed:
    seed: int


@serde
@dataclass
class ResamplingParameters:
    resample: bool
    epsilon: float
    hamiltonians: int
    split_shots: bool
    resample_single: bool


@dataclass
class OptimizationRun:
    seed: Seed
    shots: int
    steps: int
    resampling: ResamplingParameters
    optimizer: OptimizerParameters
    problem: ProblemParameters

    def get_problem(self) -> QSOProblem:
        match self.problem:
            case TightBindingParameters():
                problem: QSOProblem = TightBindingProblem(self, self.problem)

            case FeatureSelectionParameters():
                problem = FeatureSelectionProblem(self, self.problem)

        return problem


@serde(tagging=InternalTagging("kind"))
@dataclass
class OptimizationDescription:
    seed: Seed | Random | RunNumber
    shots: int
    steps: int
    resampling: ResamplingParameters
    optimizer: OptimizerParameters
    problem: ProblemParameters

    def get_run(self, run_number: int) -> OptimizationRun:
        match self.seed:
            case Seed():
                seed = self.seed
            case Random():
                seed = Seed(int.from_bytes(randbytes(4)))
            case RunNumber():
                seed = Seed(run_number)

        return OptimizationRun(seed, self.shots, self.steps, self.resampling,
                               self.optimizer, self.problem)
