from .optimizer import Optimizer
from .adam import Adam
from .spsa import SPSA
from .trust_region import AdaptiveTrustRegion

OPTIMIZER_CATALOG: dict[str, type[Optimizer]] = {
    'adam': Adam,
    'spsa': SPSA,
    'tr': AdaptiveTrustRegion,
}
