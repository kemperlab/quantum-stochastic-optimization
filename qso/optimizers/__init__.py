from .optimizer import Optimizer, Circuit, StateCircuit
from .adam import Adam, AdamParameters
from .spsa import Spsa, SpsaParameters
from .trust_region import TrustRegion, TrustRegionParameters

OptimizerParameters = AdamParameters | SpsaParameters | TrustRegionParameters
