"""Compiling a question into a search plan, and running it.

This is where the four mechanisms become one system. A plan routes a question to
the engines that can answer it, shapes a query for each, executes them
concurrently under a hard budget, and stops as soon as the evidence stops
improving — handing back whatever budget it did not need.

Planning is separated from execution on purpose, and the separation is load
bearing rather than tidy. Every planning decision here is deterministic and
costs nothing, so :meth:`Planner.plan` can enumerate every step a question could
possibly require without issuing a single search. That is what makes
:meth:`Planner.dry_run` honest: it reports the *ceiling*, the worst case where
nothing saturates early.

The gap between that ceiling and what :meth:`Planner.run` actually spends is the
project's entire claim. A dry run saying "up to 12 searches" followed by a real
run spending 7 is not an estimate that missed — it is the early stop working.

:func:`naive_run` implements the baseline the benchmark compares against: one
query, one engine, take the results. It lives here rather than in the benchmark
so that it shares this module's client and cache, which is the only way the
comparison is fair — both strategies pay the same price for the same page.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from frugal.adapters import normalise
from frugal.budget import BudgetGovernor, Projection, SpendListener
from frugal.client import SerpApiClient
from frugal.errors import BudgetExceeded, FrugalError
from frugal.parameters import Locale, build_params, detect_locale, wants_recent
from frugal.reformulate import expand_query, feedback_terms, reformulate
from frugal.router import RoutingDecision, route
from frugal.saturation import (
    DEFAULT_PATIENCE,
    DEFAULT_THRESHOLD,
    Observation,
    SaturationMonitor,
)
from frugal.schema import Document, Evidence, Series

DEFAULT_BUDGET = 12

#: Engines routed to per round, and rounds per plan.
#:
#: These were 3 and 4 until the ablation measured them. On the benchmark set, a
#: specialised engine paired with web search answers every question, while a
#: third engine and a second round buy no further recall at all -- 3x2 costs
#: 3.9 searches per question against 1.7, for identical answers.
#:
#: The pairing is what matters rather than the specialised engine alone: routing
#: to one engine scores exactly the same as plain web search, missing different
#: questions rather than fewer. Neither index covers everything and the two
#: together do.
#:
#: Stated plainly because it constrains the claim: twelve questions is a small
#: set, and one whose answers are reasonably discoverable. A harder set may well
#: pay for depth, which is why rounds remain configurable and the stopping rule
#: still governs them.
DEFAULT_MAX_ENGINES = 2
DEFAULT_MAX_ROUNDS = 1
DEFAULT_CONCURRENCY = 4

#: Results considered per search, requested from the engine and enforced after
#: normalisation.
#:
#: The enforcement matters more than the request. Engines disagree wildly about
#: page size and several ignore ``num`` entirely: one news search returns a
#: hundred headlines where a web search returns nine. Counting those equally
#: would make any engine with a generous page size look like a better plan,
#: when all it has is a longer page for the same money.
#:
#: Truncating to a common depth is standard retrieval practice and it is also
#: the honest comparison: an agent reading results has a finite context, and the
#: hundredth headline for a question was never going to be read. Without this,
#: marginal novelty stays high by construction and the stopping rule can never
#: fire.
DEFAULT_RESULTS_PER_SEARCH = 10


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One search a plan intends to issue."""

    engine: str
    query: str
    strategy: str
    cost: int
    round_number: int
    rationale: str
    #: The full request, built for this engine. Carried on the step so a plan can
    #: be inspected before it is paid for -- a missing geo is visible here rather
    #: than only in the results it quietly got wrong.
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        extra = {k: v for k, v in self.params.items() if k not in {"q", "num"}}
        detail = f" {extra}" if extra else ""
        return (
            f'round {self.round_number}: {self.engine} <- "{self.query}" '
            f"({self.strategy}){detail}"
        )


@dataclass(frozen=True, slots=True)
class ExecutedStep:
    """What a step actually did, including when it failed."""

    step: PlanStep
    evidence_count: int
    charged: int
    from_cache: bool
    elapsed_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.step.engine,
            "query": self.step.query,
            "strategy": self.step.strategy,
            "round": self.step.round_number,
            "evidence": self.evidence_count,
            "charged": self.charged,
            "from_cache": self.from_cache,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DryRun:
    """What a plan would cost, computed without spending anything."""

    question: str
    steps: tuple[PlanStep, ...]
    projection: Projection

    @property
    def ceiling(self) -> int:
        """Worst-case spend: every step issued, nothing saturating early."""
        return sum(step.cost for step in self.steps)

    def describe(self) -> str:
        engines = sorted({step.engine for step in self.steps})
        return (
            f"{len(self.steps)} steps across {len(engines)} engines "
            f"({', '.join(engines)}); at most {self.ceiling} searches, "
            f"{self.projection.describe()}"
        )


@dataclass(slots=True)
class PlanResult:
    """Everything a run produced, including why it stopped."""

    question: str
    evidence: list[Evidence]
    steps: list[ExecutedStep]
    observations: list[Observation]
    budget: Any
    saturation: Any
    stopped_because: str
    elapsed_ms: float
    ceiling: int

    @property
    def searches_charged(self) -> int:
        return sum(step.charged for step in self.steps)

    @property
    def steps_executed(self) -> int:
        """Steps the plan actually ran, cache hits included."""
        return len(self.steps)

    @property
    def steps_skipped(self) -> int:
        """Planned steps never run, because the plan stopped early.

        This is the early-stopping figure, and it is deliberately *not*
        ``ceiling - searches_charged``. That number folds in cache hits, so a
        run against a warm cache would report a large saving having stopped
        early not at all. Conflating the two would make the benchmark report
        the cache's work as the planner's.
        """
        return max(0, self.ceiling - self.steps_executed)

    @property
    def cache_hits(self) -> int:
        """Steps that ran without being billed. A separate saving from the above."""
        return sum(1 for step in self.steps if step.from_cache)

    @property
    def failures(self) -> list[ExecutedStep]:
        return [step for step in self.steps if not step.succeeded]

    def summarise(self) -> str:
        detail = f"{self.steps_executed}/{self.ceiling} planned steps"
        if self.cache_hits:
            detail += f", {self.cache_hits} from cache"
        return (
            f"{len(self.evidence)} unique results; {self.searches_charged} searches billed "
            f"({detail}) — {self.stopped_because}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "evidence_count": len(self.evidence),
            "searches_charged": self.searches_charged,
            "ceiling": self.ceiling,
            "steps_executed": self.steps_executed,
            "steps_skipped": self.steps_skipped,
            "cache_hits": self.cache_hits,
            "stopped_because": self.stopped_because,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "steps": [step.as_dict() for step in self.steps],
            "budget": self.budget.as_dict(),
            "saturation": self.saturation.as_dict(),
        }


@dataclass(slots=True)
class _RoundOutcome:
    """Internal: what one round of concurrent searches returned."""

    executed: list[ExecutedStep] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    budget_exhausted: bool = False


class Planner:
    """Turns a question into evidence, under a budget.

    :param max_engines: Engines routed to per round. More engines widen a round
        but each one costs a search, so this trades breadth against depth.
    :param max_rounds: Rounds before stopping regardless of saturation, so a
        question whose evidence never saturates still terminates.
    :param concurrency: Searches in flight at once within a round.
    """

    def __init__(
        self,
        client: SerpApiClient,
        *,
        max_engines: int = DEFAULT_MAX_ENGINES,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        concurrency: int = DEFAULT_CONCURRENCY,
        results_per_search: int = DEFAULT_RESULTS_PER_SEARCH,
        saturation_threshold: float | None = None,
        saturation_patience: int | None = None,
        force_engines: tuple[str, ...] | None = None,
    ) -> None:
        if max_engines < 1:
            raise ValueError(f"max_engines must be at least 1, got {max_engines}")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")

        self.client = client
        # An experimental control, not a feature: fixing the engines bypasses
        # routing entirely, which is how the benchmark measures whether routing
        # is doing the work or whether any second engine would serve as well.
        self.force_engines = force_engines
        self.max_engines = max_engines
        self.max_rounds = max_rounds
        self.concurrency = concurrency
        self.results_per_search = results_per_search
        self._threshold = (
            DEFAULT_THRESHOLD if saturation_threshold is None else saturation_threshold
        )
        self._patience = DEFAULT_PATIENCE if saturation_patience is None else saturation_patience

    # -- planning, which costs nothing -------------------------------------

    def locale_for(self, question: str) -> Locale | None:
        """The place a question is about, detected once per plan."""
        return detect_locale(question)

    def plan(self, question: str) -> list[PlanStep]:
        """Enumerate every step this question could require.

        Deterministic and free: routing and reformulation both run without
        touching the network, so the full plan can be inspected before deciding
        whether to pay for it. Reformulation is threaded through rounds exactly
        as execution would thread it, so a step appears here only if execution
        would genuinely issue it — a plan that runs out of distinct queries
        stops here too.
        """
        # Detected from the question rather than from a reformulated query: a
        # variant may have dropped the place name that identifies the locale,
        # and the plan still needs to know where the question is about.
        locale = self.locale_for(question)
        recent = wants_recent(question)
        year = datetime.now(UTC).year

        if self.force_engines is not None:
            decisions = [
                RoutingDecision(engine=name, score=0.0, cost=1, reasons=("fixed",))
                for name in self.force_engines[: self.max_engines]
            ]
        else:
            decisions = route(question, limit=self.max_engines)
        issued: dict[str, tuple[str, ...]] = {d.engine: () for d in decisions}
        steps: list[PlanStep] = []

        for round_number in range(1, self.max_rounds + 1):
            produced = 0
            for decision in decisions:
                variants = reformulate(
                    question,
                    engine=decision.engine,
                    issued=issued[decision.engine],
                    limit=1,
                )
                if not variants:
                    continue
                variant = variants[0]
                steps.append(
                    PlanStep(
                        engine=decision.engine,
                        query=variant.query,
                        strategy=variant.strategy,
                        cost=decision.cost,
                        round_number=round_number,
                        rationale=variant.rationale,
                        params=build_params(
                            decision.engine,
                            variant.query,
                            locale=locale,
                            results=self.results_per_search,
                            recent=recent,
                            current_year=year,
                        ),
                    )
                )
                issued[decision.engine] += (variant.query,)
                produced += 1

            if not produced:
                # Every engine has run out of distinct angles. More rounds would
                # only reissue queries the plan has already made.
                break

        return steps

    def dry_run(self, question: str, *, budget: int = DEFAULT_BUDGET) -> DryRun:
        """Report what a plan would cost without issuing anything."""
        steps = self.plan(question)
        governor = BudgetGovernor(limit=budget)
        return DryRun(
            question=question,
            steps=tuple(steps),
            projection=governor.project([step.cost for step in steps]),
        )

    # -- execution ---------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        budget: int = DEFAULT_BUDGET,
        listener: SpendListener | None = None,
    ) -> PlanResult:
        """Execute a plan, stopping as soon as the evidence stops improving."""
        if not question or not question.strip():
            raise ValueError("cannot plan for an empty question")

        started = time.perf_counter()
        governor = BudgetGovernor(limit=budget, listener=listener)
        monitor = SaturationMonitor(threshold=self._threshold, patience=self._patience)

        planned = self.plan(question)
        ceiling = sum(step.cost for step in planned)
        by_round: dict[int, list[PlanStep]] = {}
        for step in planned:
            by_round.setdefault(step.round_number, []).append(step)

        executed: list[ExecutedStep] = []
        observations: list[Observation] = []
        evidence: list[Evidence] = []
        stopped = "completed the plan"

        locale = self.locale_for(question)

        for round_number in sorted(by_round):
            if governor.exhausted:
                stopped = "budget exhausted"
                break

            steps = by_round[round_number]
            if round_number > 1:
                # Rounds after the first adapt to what the earlier ones found,
                # rather than rewording the question again.
                steps = self._adapt(steps, question, evidence, locale)

            outcome = self._run_round(steps, governor)
            executed.extend(outcome.executed)

            observation = monitor.observe(outcome.evidence)
            observations.append(observation)
            # observe() has already recorded these identities, so the monitor
            # cannot filter them for us; dedupe against what is collected.
            evidence = _dedupe_preserving_order(evidence + outcome.evidence)

            if monitor.saturated:
                stopped = "stopped early: evidence saturated"
                break

            if outcome.budget_exhausted:
                stopped = "budget exhausted"
                break

        return PlanResult(
            question=question,
            evidence=evidence,
            steps=executed,
            observations=observations,
            budget=governor.report(),
            saturation=monitor.report(),
            stopped_because=stopped,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            ceiling=ceiling,
        )

    def _adapt(
        self,
        steps: list[PlanStep],
        question: str,
        evidence: list[Evidence],
        locale: Locale | None,
    ) -> list[PlanStep]:
        """Rewrite a round's steps using vocabulary the earlier rounds retrieved.

        Pseudo-relevance feedback: the results so far are treated as relevant and
        mined for terms the question never contained. A question about "the
        latest ISRO mission" does not name the mission; a later round that does
        is asking something genuinely new rather than rephrasing.

        Deterministic, so a plan that adapts to what it found still reproduces
        exactly. Term-only engines are left alone: ``google_trends`` matches a
        short term against its index, and appending a feedback term to that
        produces a term nobody has searched for.
        """
        terms = feedback_terms(
            [_feedback_text(item) for item in evidence], question, limit=2
        )
        if not terms:
            return steps

        adapted: list[PlanStep] = []
        for step in steps:
            if step.engine in _TERM_ONLY:
                adapted.append(step)
                continue
            expanded = expand_query(step.query, terms)
            if expanded == step.query:
                adapted.append(step)
                continue
            adapted.append(
                PlanStep(
                    engine=step.engine,
                    query=expanded,
                    strategy=f"{step.strategy}+feedback",
                    cost=step.cost,
                    round_number=step.round_number,
                    rationale=f"expanded with terms found earlier: {', '.join(terms)}",
                    params=build_params(
                        step.engine,
                        expanded,
                        locale=locale,
                        results=self.results_per_search,
                    ),
                )
            )
        return adapted

    def _run_round(self, steps: list[PlanStep], governor: BudgetGovernor) -> _RoundOutcome:
        """Execute one round's steps concurrently."""
        outcome = _RoundOutcome()
        workers = min(self.concurrency, len(steps))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._execute, step, governor): step for step in steps}
            for future in futures:
                step_result, produced, exhausted = future.result()
                outcome.executed.append(step_result)
                outcome.evidence.extend(produced)
                outcome.budget_exhausted = outcome.budget_exhausted or exhausted

        # Ordered by the plan rather than by completion, so a run is reproducible
        # regardless of which engine happened to answer first.
        outcome.executed.sort(key=lambda e: steps.index(e.step))
        return outcome

    def _execute(
        self, step: PlanStep, governor: BudgetGovernor
    ) -> tuple[ExecutedStep, list[Evidence], bool]:
        """Run one step, charging the budget only for what SerpApi billed."""
        started = time.perf_counter()
        params = dict(step.params)

        # Establish that the search is free before reserving for it. Reserving
        # first means an exhausted budget can refuse a cache hit that would have
        # cost nothing, which is the opposite of what a budget is for.
        cached = self.client.cached(step.engine, **params)
        if cached is not None:
            return self._finish(step, cached, started, charged=False)

        try:
            claim = governor.open_reservation(step.cost, engine=step.engine)
        except BudgetExceeded:
            return (
                ExecutedStep(
                    step=step,
                    evidence_count=0,
                    charged=0,
                    from_cache=False,
                    elapsed_ms=0.0,
                    error="budget exhausted before this step",
                ),
                [],
                True,
            )

        try:
            response = self.client.search(step.engine, **params)
        except FrugalError as exc:
            # A failed engine must not take the plan down with it: the other
            # engines in this round may still answer the question.
            claim.release()
            return (
                ExecutedStep(
                    step=step,
                    evidence_count=0,
                    charged=0,
                    from_cache=False,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                [],
                False,
            )

        # A concurrent step may have recorded this same search since the peek.
        if response.from_cache:
            claim.release()
        else:
            claim.commit()

        return self._finish(step, response, started, charged=not response.from_cache)

    def _finish(
        self, step: PlanStep, response: Any, started: float, *, charged: bool
    ) -> tuple[ExecutedStep, list[Evidence], bool]:
        """Normalise a response into evidence and account for the step."""
        elapsed = (time.perf_counter() - started) * 1000
        billed = response.searches_charged if charged else 0

        try:
            produced = normalise(
                step.engine,
                response.raw,
                query=step.query,
                search_id=response.search_id,
            )
        except FrugalError as exc:
            # The search was billed; the shape changed. Report it rather than
            # silently returning nothing, which would look like a thin engine.
            return (
                ExecutedStep(
                    step=step,
                    evidence_count=0,
                    charged=billed,
                    from_cache=response.from_cache,
                    elapsed_ms=elapsed,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                [],
                False,
            )

        # Enforced here rather than trusted to the engine: several ignore `num`.
        produced = produced[: self.results_per_search]

        return (
            ExecutedStep(
                step=step,
                evidence_count=len(produced),
                charged=billed,
                from_cache=response.from_cache,
                elapsed_ms=elapsed,
            ),
            produced,
            False,
        )


def naive_run(
    client: SerpApiClient,
    question: str,
    *,
    results: int = DEFAULT_RESULTS_PER_SEARCH,
) -> PlanResult:
    """The baseline: one query, one engine, take the results.

    This is what most agents do, and it is implemented fairly rather than as a
    strawman — it gets the same client, the same cache and the same page size,
    so both strategies pay the same price for the same page. The only difference
    measured is the planning.
    """
    started = time.perf_counter()
    governor = BudgetGovernor(limit=1)
    monitor = SaturationMonitor()
    step = PlanStep(
        engine="google",
        query=question.strip(),
        strategy="verbatim",
        cost=1,
        round_number=1,
        rationale="baseline: the question as asked, sent to web search",
        # Deliberately unparameterised beyond page size. Adding geo or location
        # here would be crediting the baseline with the engine knowledge the
        # planner exists to supply, and the comparison would stop meaning
        # anything.
        params={"q": question.strip(), "num": results},
    )

    evidence: list[Evidence] = []
    error: str | None = None
    charged = 0
    from_cache = False

    try:
        with governor.reserve(engine="google") as claim:
            response = client.search("google", q=step.query, num=results)
            if response.from_cache:
                claim.release()
            charged = response.searches_charged
            from_cache = response.from_cache
            evidence = normalise("google", response.raw, query=step.query)[:results]
    except FrugalError as exc:
        error = f"{type(exc).__name__}: {exc}"

    observation = monitor.observe(evidence)

    return PlanResult(
        question=question,
        evidence=evidence,
        steps=[
            ExecutedStep(
                step=step,
                evidence_count=len(evidence),
                charged=charged,
                from_cache=from_cache,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
        ],
        observations=[observation],
        budget=governor.report(),
        saturation=monitor.report(),
        stopped_because="baseline: a single search",
        elapsed_ms=(time.perf_counter() - started) * 1000,
        ceiling=1,
    )


def keyword_naive_run(
    client: SerpApiClient,
    question: str,
    *,
    results: int = DEFAULT_RESULTS_PER_SEARCH,
) -> PlanResult:
    """A stronger baseline: one web search, but with the question reformulated.

    The verbatim baseline sends a natural-language sentence to a keyword engine,
    and on some questions that returns nonsense -- "Where are the major hospitals
    in Coimbatore?" surfaced a music video called "Major". That is what many
    agents actually do, so it is worth measuring, but on its own it makes the
    planner look good for the wrong reason.

    This baseline gets the planner's own keyword reformulation and nothing else:
    same single search, same engine, no routing, no locale parameters. The gap
    between this and the verbatim baseline is what reformulation is worth; the
    gap between this and a full plan is what routing is worth. Reporting only the
    weaker baseline would credit routing with reformulation's work.
    """
    started = time.perf_counter()
    governor = BudgetGovernor(limit=1)
    monitor = SaturationMonitor()

    variants = reformulate(question, engine="google", limit=1)
    query = variants[0].query if variants else question.strip()

    step = PlanStep(
        engine="google",
        query=query,
        strategy="keyword",
        cost=1,
        round_number=1,
        rationale="baseline: one web search, question reformulated to keywords",
        params={"q": query, "num": results},
    )

    evidence: list[Evidence] = []
    error: str | None = None
    charged = 0
    from_cache = False

    try:
        with governor.reserve(engine="google") as claim:
            response = client.search("google", q=query, num=results)
            if response.from_cache:
                claim.release()
            charged = response.searches_charged
            from_cache = response.from_cache
            evidence = normalise("google", response.raw, query=query)[:results]
    except FrugalError as exc:
        error = f"{type(exc).__name__}: {exc}"

    observation = monitor.observe(evidence)

    return PlanResult(
        question=question,
        evidence=evidence,
        steps=[
            ExecutedStep(
                step=step,
                evidence_count=len(evidence),
                charged=charged,
                from_cache=from_cache,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
        ],
        observations=[observation],
        budget=governor.report(),
        saturation=monitor.report(),
        stopped_because="baseline: a single reformulated search",
        elapsed_ms=(time.perf_counter() - started) * 1000,
        ceiling=1,
    )


#: Engines that match a short term rather than a phrase, and so must not have
#: feedback terms appended to their queries.
_TERM_ONLY = frozenset({"google_trends"})


def _feedback_text(item: Evidence) -> str:
    """The text of one result, for mining vocabulary out of."""
    if isinstance(item, Series):
        return item.name
    if isinstance(item, Document):
        return " ".join(part for part in (item.title, item.snippet, item.source) if part)
    return ""  # pragma: no cover - Evidence is a closed union


def _dedupe_preserving_order(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    unique: list[Evidence] = []
    for item in evidence:
        if item.identity in seen:
            continue
        seen.add(item.identity)
        unique.append(item)
    return unique
