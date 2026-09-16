"""Exception hierarchy.

Every error Frugal raises derives from :class:`FrugalError`, so callers embedding
the planner can catch one type and never see a bare library exception escape.
"""

from __future__ import annotations


class FrugalError(Exception):
    """Base class for every error raised by Frugal."""


class CacheMiss(FrugalError):
    """A key was absent from the cache while in a mode that cannot fetch.

    Raised only in :attr:`~frugal.cache.CacheMode.REPLAY`. A miss during REPLAY
    means the recorded fixture set is incomplete for the workload being run,
    which would silently invalidate a benchmark — so it is an error, not a
    fall-through to the network.
    """

    def __init__(self, engine: str, key: str) -> None:
        self.engine = engine
        self.key = key
        super().__init__(
            f"no recorded response for engine={engine!r} key={key[:12]}... "
            f"(REPLAY mode cannot fetch; re-record this workload)"
        )


class CacheCorrupt(FrugalError):
    """A cache entry exists on disk but could not be decoded."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"corrupt cache entry at {path}: {reason}")


class BudgetExceeded(FrugalError):
    """A plan tried to spend more searches than its budget allows."""

    def __init__(self, spent: int, limit: int) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"search budget exhausted: {spent}/{limit} searches spent")


class TransportError(FrugalError):
    """The SerpApi request failed after exhausting retries."""
