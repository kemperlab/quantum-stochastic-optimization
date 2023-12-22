from .optimizer import Optimizer
from .adam import Adam
from .spsa import SPSA
from .trust_region import AdaptiveTrustRegion

OPTIMIZER_CATALOG: dict[str, type[Optimizer]] = {
    'adam': Adam,
    'spsa': SPSA,
    'tr': AdaptiveTrustRegion,
}


def get_optimizer(name: str) -> type[Optimizer]:
    if name in OPTIMIZER_CATALOG:
        return OPTIMIZER_CATALOG[name]
    else:
        raise ValueError(f"Could not find optimizer `{name}`")
