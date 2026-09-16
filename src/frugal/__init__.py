"""Frugal: a cost-aware search planner for SerpApi.

Frugal compiles a question into a *search plan* — which engines to call, in what
order, with which query reformulations — executed under an explicit budget with
early stopping once the retrieved evidence stops improving.
"""

from frugal.errors import (
    BudgetExceeded,
    CacheCorrupt,
    CacheMiss,
    FrugalError,
    TransportError,
)

__version__ = "0.1.0"

__all__ = [
    "BudgetExceeded",
    "CacheCorrupt",
    "CacheMiss",
    "FrugalError",
    "TransportError",
    "__version__",
]
