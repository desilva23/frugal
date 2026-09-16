"""Knowing when to stop searching.

Most agents have no stopping rule. They search a fixed number of times, or until
a model declares itself finished, and in both cases they routinely keep paying
for pages they have already seen. A plan that has found its answer should stop
and hand the remaining budget back.

The signal is marginal novelty: of the results this batch returned, how many
were evidence the plan did not already have. A first batch is all new by
definition. As a plan exhausts the genuinely distinct sources for a question,
that fraction falls, and once it has stayed low for a couple of batches there is
nothing left to buy.

Patience matters. One thin batch is ordinary — a reformulation that happened to
land badly, an engine with little coverage of the topic — and stopping on it
would abandon a plan that had more to find. Requiring consecutive thin batches
distinguishes a plan that has run dry from one that hit a bad query.

This monitor also owns the deduplication set, which is deliberate: "how much new
evidence arrived" and "which results are duplicates" are the same question, and
answering it twice in two places invites the two answers to disagree.

The cumulative curve it records — unique evidence against searches spent — is
the benchmark's headline figure. Naive and planned search plotted on the same
axes is the clearest statement of what this project does.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from frugal.schema import Evidence

#: Marginal novelty below which a batch counts as thin. A fifth of a batch being
#: new still means four fifths of that search bought nothing.
DEFAULT_THRESHOLD = 0.2

#: Consecutive thin batches before a plan stops. One is a bad query; two is a
#: pattern.
DEFAULT_PATIENCE = 2

#: Batches observed before stopping is permitted at all, however thin they are.
#: A plan that stopped after one batch never gave a second engine a chance to
#: contribute, which is the entire point of routing across several.
DEFAULT_MIN_BATCHES = 2


@dataclass(frozen=True, slots=True)
class Observation:
    """What one batch of results added, and whether the plan should continue."""

    batch_size: int
    new_items: int
    marginal_novelty: float
    cumulative_unique: int
    searches_so_far: int
    saturated: bool
    reason: str

    def explain(self) -> str:
        return (
            f"batch {self.searches_so_far}: {self.new_items}/{self.batch_size} new "
            f"({self.marginal_novelty:.0%}), {self.cumulative_unique} unique so far — {self.reason}"
        )


@dataclass(slots=True)
class SaturationReport:
    """The evidence-against-effort curve, for the benchmark and the trace."""

    searches: int
    unique_items: int
    curve: tuple[int, ...]
    marginal: tuple[float, ...]
    saturated: bool
    stopped_at: int | None

    @property
    def efficiency(self) -> float:
        """Unique evidence per search. The number the benchmark compares."""
        return self.unique_items / self.searches if self.searches else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "searches": self.searches,
            "unique_items": self.unique_items,
            "efficiency": round(self.efficiency, 4),
            "curve": list(self.curve),
            "marginal": [round(m, 4) for m in self.marginal],
            "saturated": self.saturated,
            "stopped_at": self.stopped_at,
        }


class SaturationMonitor:
    """Tracks whether a plan is still learning anything.

    Feed it each batch of normalised evidence as it arrives; ask
    :attr:`saturated` before spending again.

    .. code-block:: python

        monitor = SaturationMonitor()
        for query in queries:
            monitor.observe(search(query))
            if monitor.saturated:
                break
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        patience: int = DEFAULT_PATIENCE,
        min_batches: int = DEFAULT_MIN_BATCHES,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if min_batches < 1:
            raise ValueError(f"min_batches must be at least 1, got {min_batches}")

        self.threshold = threshold
        self.patience = patience
        self.min_batches = min_batches

        self._seen: set[str] = set()
        self._curve: list[int] = []
        self._marginal: list[float] = []
        self._thin_run = 0
        self._saturated = False
        self._stopped_at: int | None = None

    # -- state ------------------------------------------------------------

    @property
    def saturated(self) -> bool:
        """Whether the plan should stop searching."""
        return self._saturated

    @property
    def unique_count(self) -> int:
        return len(self._seen)

    @property
    def batches(self) -> int:
        return len(self._curve)

    def has_seen(self, item: Evidence) -> bool:
        return item.identity in self._seen

    # -- observation ------------------------------------------------------

    def observe(self, evidence: Iterable[Evidence]) -> Observation:
        """Record one batch and report what it added.

        Duplicates *within* a batch count once: two engines returning the same
        page in one round is one piece of evidence, not two.
        """
        batch = list(evidence)
        identities = list(dict.fromkeys(item.identity for item in batch))
        fresh = [key for key in identities if key not in self._seen]
        self._seen.update(fresh)

        # An empty batch is not novel. Dividing by its size would raise, and
        # treating it as fully novel would keep a dead plan alive.
        marginal = len(fresh) / len(batch) if batch else 0.0

        self._curve.append(len(self._seen))
        self._marginal.append(marginal)

        if marginal < self.threshold:
            self._thin_run += 1
        else:
            # A productive batch means the plan is still learning; earlier thin
            # ones were bad queries rather than an exhausted question.
            self._thin_run = 0

        reason = self._decide()

        return Observation(
            batch_size=len(batch),
            new_items=len(fresh),
            marginal_novelty=round(marginal, 4),
            cumulative_unique=len(self._seen),
            searches_so_far=len(self._curve),
            saturated=self._saturated,
            reason=reason,
        )

    def _decide(self) -> str:
        """Update saturation and return the reason, in words."""
        if self._saturated:
            return "already stopped"

        if len(self._curve) < self.min_batches:
            return f"too early to stop ({len(self._curve)}/{self.min_batches} batches)"

        if self._thin_run >= self.patience:
            self._saturated = True
            self._stopped_at = len(self._curve)
            return (
                f"saturated: {self._thin_run} consecutive batches below "
                f"{self.threshold:.0%} new evidence"
            )

        if self._thin_run:
            return f"thin batch ({self._thin_run}/{self.patience} before stopping)"
        return "still finding new evidence"

    def unique(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        """Filter ``evidence`` to items not yet seen, without recording them.

        Lets a caller inspect what a batch would add before deciding to keep it.
        """
        return [item for item in evidence if item.identity not in self._seen]

    def report(self) -> SaturationReport:
        return SaturationReport(
            searches=len(self._curve),
            unique_items=len(self._seen),
            curve=tuple(self._curve),
            marginal=tuple(self._marginal),
            saturated=self._saturated,
            stopped_at=self._stopped_at,
        )

    def __repr__(self) -> str:
        return (
            f"SaturationMonitor(batches={self.batches}, unique={self.unique_count}, "
            f"saturated={self._saturated})"
        )
