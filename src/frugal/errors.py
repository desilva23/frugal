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


class MissingAPIKey(FrugalError):
    """No SerpApi credential could be resolved.

    The message is deliberately a set of instructions rather than a statement of
    fact: this is the first error most people meet, and it should tell them what
    to do rather than what went wrong.
    """

    def __init__(self, searched: list[str] | None = None) -> None:
        self.searched = searched or []
        locations = "\n".join(f"  - {s}" for s in self.searched)
        super().__init__(
            "No SerpApi key found.\n\n"
            "Create a .env file in the project root containing:\n\n"
            "    SERPAPI_API_KEY=your-key-here\n\n"
            "Get a free key (250 searches/month) at https://serpapi.com/manage-api-key\n"
            + (f"\nLooked in:\n{locations}" if self.searched else "")
        )


class SerpApiError(FrugalError):
    """SerpApi accepted the request but reported an error for the search.

    SerpApi signals most search-level problems with HTTP 200 and an ``error``
    field in the body rather than a failure status, so this is raised from
    response inspection rather than from the transport.
    """

    def __init__(self, engine: str, message: str, *, status: str | None = None) -> None:
        self.engine = engine
        self.api_message = message
        self.status = status
        detail = f" (status={status})" if status else ""
        super().__init__(f"{engine}: {message}{detail}")


class AuthenticationError(FrugalError):
    """The SerpApi key was rejected.

    Never retried: a rejected key will stay rejected, and retrying burns time
    and obscures the real problem.
    """


class RateLimited(FrugalError):
    """SerpApi returned 429 and retries did not clear it."""

    def __init__(self, attempts: int, retry_after: float | None = None) -> None:
        self.attempts = attempts
        self.retry_after = retry_after
        hint = f"; server asked for {retry_after:.0f}s" if retry_after else ""
        super().__init__(f"rate limited by SerpApi after {attempts} attempts{hint}")
