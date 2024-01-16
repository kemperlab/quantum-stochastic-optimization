from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from random import randbytes
from serde import serde, InternalTagging

from .tight_binding import TightBindingProblem
from .tight_binding.params import TightBindingParameters
from .feature_selection import FeatureSelectionProblem
from .feature_selection.params import FeatureSelectionParameters
from .resampling_params import ResamplingParameters

from ..optimizers import OptimizerParameters

ProblemParameters = TightBindingParameters | FeatureSelectionParameters

if TYPE_CHECKING:
    from .problem import QSOProblem


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
