from dataclasses import dataclass
from serde import serde


@serde
@dataclass
class MultipleHamiltonians:
    epsilon: float
    hamiltonians: int
    split_shots: bool


@serde
@dataclass
class ExpectedHamiltonian:
    hamiltonians: int


@serde
@dataclass
class DefaultHamiltonian:
    pass


ResamplingParameters = MultipleHamiltonians | ExpectedHamiltonian | DefaultHamiltonian
