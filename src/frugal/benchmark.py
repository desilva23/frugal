"""Measuring whether planned search finds answers, and what it costs.

The metric here is answer recall, not result count. An earlier version of this
comparison reported unique results per search, and running it honestly showed
the two strategies were indistinguishable on that measure — planned search
returned 8.7 unique results per search against the baseline's 9.0. That number
was never the claim. Ten news headlines are not ten pieces of evidence, and a
paper that answers a research question outweighs a page that mentions it.

So a question is scored on whether the evidence retrieved actually contains its
answer. Each question carries marker groups: every group must be satisfied, and
a group is satisfied by any of its alternatives, so a correct answer phrased
differently still counts. Scoring is exact string matching, which means it is
free, deterministic, and reproducible by anyone — no judge model, nothing to
take on trust.

Depth is recorded alongside recall: how far into the evidence a reader would
have to go before the answer was complete. Two strategies that both find an
answer are not equal if one puts it first and the other puts it fifteenth.

Run it in replay mode and the whole benchmark re-executes from committed
fixtures at zero cost, which is the property that makes the published table
checkable rather than merely stated.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.errors import FrugalError
from frugal.planner import (
    DEFAULT_BUDGET,
    Planner,
    PlanResult,
    keyword_naive_run,
    naive_run,
)
from frugal.schema import Evidence, Series

QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "results.json"

#: Three strategies, because two could not tell routing from reformulation.
#: "naive" sends the question verbatim, which is what many agents do.
#: "keyword" sends the planner's own reformulation to one engine, isolating
#: what the reformulation is worth. "planned" adds routing on top.
STRATEGIES = ("naive", "keyword", "planned")


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question and the evidence that would answer it."""

    id: str
    question: str
    category: str
    markers: tuple[tuple[str, ...], ...]
    rationale: str
    expects_series: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Question:
        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            category=str(raw.get("category", "uncategorised")),
            markers=tuple(tuple(str(a).lower() for a in group) for group in raw["markers"]),
            rationale=str(raw.get("rationale", "")),
            expects_series=bool(raw.get("expects_series", False)),
        )


@dataclass(frozen=True, slots=True)
class Score:
    """Whether a question was answered, and how deep the answer was."""

    found: bool
    groups_matched: int
    groups_total: int
    depth: int | None
    missing: tuple[str, ...]
    #: Whether the evidence included a time series.
    structured: bool = False
    #: Whether the question is one a series answers better than prose.
    wanted_structured: bool = False

    @property
    def answered_in_the_right_modality(self) -> bool:
        """Answered with structured data where the question called for it.

        Reported beside recall rather than inside it. "Interest rose 120%" in an
        article and a 53-point series both answer the question; only one of them
        can be plotted, compared, or checked for when the change happened.
        """
        return not self.wanted_structured or self.structured

    @property
    def completeness(self) -> float:
        """Fraction of marker groups satisfied. Partial credit, for diagnosis."""
        return self.groups_matched / self.groups_total if self.groups_total else 0.0


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """One strategy's attempt at one question."""

    question_id: str
    category: str
    strategy: str
    score: Score
    searches: int
    billed_this_run: int
    evidence_count: int
    engines_used: tuple[str, ...]
    elapsed_ms: float
    stopped_because: str
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "strategy": self.strategy,
            "found": self.score.found,
            "completeness": round(self.score.completeness, 3),
            "depth": self.score.depth,
            "structured": self.score.structured,
            "right_modality": self.score.answered_in_the_right_modality,
            "missing": list(self.score.missing),
            "searches": self.searches,
            "billed_this_run": self.billed_this_run,
            "evidence_count": self.evidence_count,
            "engines_used": list(self.engines_used),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "stopped_because": self.stopped_because,
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class StrategySummary:
    """Aggregate performance of one strategy across the question set."""

    strategy: str
    questions: int
    answered: int
    searches: int
    depths: list[int] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.answered / self.questions if self.questions else 0.0

    @property
    def searches_per_question(self) -> float:
        return self.searches / self.questions if self.questions else 0.0

    @property
    def answers_per_search(self) -> float:
        """The efficiency figure: answers found per search bought."""
        return self.answered / self.searches if self.searches else 0.0

    @property
    def median_depth(self) -> float | None:
        """Typical position of the completed answer. Lower is better."""
        if not self.depths:
            return None
        ordered = sorted(self.depths)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "questions": self.questions,
            "answered": self.answered,
            "recall": round(self.recall, 4),
            "searches": self.searches,
            "searches_per_question": round(self.searches_per_question, 2),
            "answers_per_search": round(self.answers_per_search, 4),
            "median_depth": self.median_depth,
        }


def searchable_text(item: Evidence) -> str:
    """Everything about one result a marker could reasonably match.

    A series contributes its name and its summary line, because "trending up
    11%" is the evidence for a question about direction of change and there is
    no document to match against.
    """
    if isinstance(item, Series):
        return f"{item.name} {item.summarise()}".lower()

    parts: list[str] = [item.title]
    for value in (item.snippet, item.source, item.url):
        if value:
            parts.append(value)
    for value in item.extra.values():
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            parts.append(str(value))
    return " ".join(parts).lower()


def score_evidence(question: Question, evidence: Sequence[Evidence]) -> Score:
    """Score retrieved evidence against a question's markers.

    Walks the evidence in rank order and records the position at which the last
    outstanding group was satisfied, so depth reflects how far a reader would
    have had to go rather than merely whether the answer was present somewhere.
    """
    # expects_series deliberately does NOT gate recall.
    #
    # It used to. Requiring an isinstance(item, Series) made it arithmetically
    # impossible for a web-search baseline to answer the trend question, since
    # web search returns documents and never a series -- and the baseline's
    # results plainly did answer it, carrying headlines like "Why India is
    # Seeing EV Interest Rise". Scoring that as a miss inflated the headline
    # from a one-question gap to a two-question one, which is an engineered
    # metric however sincerely it was arrived at.
    #
    # Whether a strategy reached structured data is still worth knowing, so it
    # is reported alongside recall rather than folded into it.
    total = len(question.markers)

    satisfied: set[int] = set()
    depth: int | None = None
    has_series = False

    for position, item in enumerate(evidence, start=1):
        text = searchable_text(item)

        for index, group in enumerate(question.markers):
            if index not in satisfied and any(alt in text for alt in group):
                satisfied.add(index)

        if isinstance(item, Series):
            has_series = True

        if depth is None and len(satisfied) == total:
            depth = position

    missing: list[str] = [
        " | ".join(group) for index, group in enumerate(question.markers) if index not in satisfied
    ]

    return Score(
        found=len(satisfied) == total,
        groups_matched=len(satisfied),
        groups_total=total,
        depth=depth,
        missing=tuple(missing),
        structured=has_series,
        wanted_structured=question.expects_series,
    )


def _outcome(
    question: Question, strategy: str, result: PlanResult
) -> QuestionOutcome:
    return QuestionOutcome(
        question_id=question.id,
        category=question.category,
        strategy=strategy,
        score=score_evidence(question, result.evidence),
        # Steps executed, not searches billed: billing depends on how warm the
        # cache happened to be, so reporting it would make a replayed benchmark
        # show every strategy as free.
        searches=result.steps_executed,
        billed_this_run=result.searches_charged,
        evidence_count=len(result.evidence),
        engines_used=tuple(dict.fromkeys(s.step.engine for s in result.steps)),
        elapsed_ms=result.elapsed_ms,
        stopped_because=result.stopped_because,
        errors=tuple(s.error for s in result.steps if s.error),
    )


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read the question set, failing loudly on a malformed file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FrugalError(f"no question set at {path}") from None
    except json.JSONDecodeError as exc:
        raise FrugalError(f"question set at {path} is not valid JSON: {exc}") from exc

    raw = payload.get("questions")
    if not isinstance(raw, list) or not raw:
        raise FrugalError(f"question set at {path} contains no questions")

    questions = [Question.from_dict(entry) for entry in raw]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise FrugalError("question ids must be unique")
    return questions


def run_benchmark(
    client: SerpApiClient,
    questions: Sequence[Question],
    *,
    budget: int = DEFAULT_BUDGET,
    planner: Planner | None = None,
    verbose: bool = True,
) -> list[QuestionOutcome]:
    """Run both strategies over every question."""
    planner = planner or Planner(client)
    outcomes: list[QuestionOutcome] = []

    for index, question in enumerate(questions, start=1):
        if verbose:
            print(f"\n[{index}/{len(questions)}] {question.id}: {question.question}")

        for strategy in STRATEGIES:
            try:
                if strategy == "naive":
                    result = naive_run(client, question.question)
                elif strategy == "keyword":
                    result = keyword_naive_run(client, question.question)
                else:
                    result = planner.run(question.question, budget=budget)
            except FrugalError as exc:
                # One strategy failing on one question must not lose the rest of
                # the run, which may have cost real searches to get this far.
                if verbose:
                    print(f"    {strategy:<8} FAILED  {type(exc).__name__}: {exc}")
                continue

            outcome = _outcome(question, strategy, result)
            outcomes.append(outcome)

            if verbose:
                mark = "FOUND" if outcome.score.found else " miss"
                depth = f"@{outcome.score.depth}" if outcome.score.depth else "  -"
                print(
                    f"    {strategy:<8} {mark} {depth:<5} "
                    f"{outcome.searches} searches, "
                    f"{outcome.evidence_count} results, "
                    f"{outcome.score.groups_matched}/{outcome.score.groups_total} markers"
                )

    return outcomes


def summarise(outcomes: Sequence[QuestionOutcome]) -> dict[str, StrategySummary]:
    """Aggregate outcomes per strategy."""
    summaries: dict[str, StrategySummary] = {}
    for outcome in outcomes:
        summary = summaries.setdefault(
            outcome.strategy,
            StrategySummary(strategy=outcome.strategy, questions=0, answered=0, searches=0),
        )
        summary.questions += 1
        summary.answered += int(outcome.score.found)
        summary.searches += outcome.searches
        if outcome.score.depth is not None:
            summary.depths.append(outcome.score.depth)
    return summaries


def render_table(summaries: dict[str, StrategySummary]) -> str:
    """The headline table, in Markdown, ready for the README."""
    header = (
        "| strategy | answered | recall | searches | searches/question "
        "| answers/search | median depth |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for strategy in STRATEGIES:
        summary = summaries.get(strategy)
        if summary is None:
            continue
        depth = "—" if summary.median_depth is None else f"{summary.median_depth:g}"
        rows.append(
            f"| {summary.strategy} | {summary.answered}/{summary.questions} "
            f"| {summary.recall:.0%} | {summary.searches} "
            f"| {summary.searches_per_question:.1f} "
            f"| {summary.answers_per_search:.2f} | {depth} |"
        )
    return "\n".join([header, *rows])


#: Plan sizes swept by ``--ablate``. Varying the plan rather than the budget is
#: deliberate: a budget limits *spend*, and cached searches are free, so a budget
#: sweep against a warm cache would not constrain anything. Plan size constrains
#: the work itself and so produces the same curve however warm the cache is.
ABLATIONS: tuple[tuple[str, int, int], ...] = (
    ("routed-1x1", 1, 1),
    ("routed-2x1", 2, 1),
    ("routed-3x1", 3, 1),
    ("routed-3x2", 3, 2),
)

#: Fixed pairings: web search plus the same second engine for every question,
#: whatever the question is about. These cost exactly what routed-2x1 costs, so
#: they answer the only question that matters about the routing -- whether it is
#: doing the work, or whether any second engine would serve as well. If a fixed
#: pairing matched routed-2x1, the router would be decoration.
FIXED_PAIRINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fixed-news", ("google", "google_news")),
    ("fixed-scholar", ("google", "google_scholar")),
    ("fixed-shopping", ("google", "google_shopping")),
)


#: Plan sizes whose cost grows fastest with the question set: a third engine
#: and a second round each add a search per question. Measured at twelve
#: questions, where both bought no recall at all, and skippable at thirty so a
#: sweep does not cost more than the benchmark it is checking.
DEEP_ABLATIONS = frozenset({"routed-3x1", "routed-3x2"})


def run_ablation(
    client: SerpApiClient,
    questions: Sequence[Question],
    *,
    budget: int,
    verbose: bool = True,
    skip_deep: bool = False,
) -> dict[str, StrategySummary]:
    """Sweep plan size, to show what each additional search buys.

    A single comparison of naive against a full plan answers "is it better?" but
    not "by how much, for what?". The curve separates the recall that comes from
    routing to the right engine at all -- which costs one search -- from the
    recall that comes from searching the same engines harder.
    """
    summaries: dict[str, StrategySummary] = {}

    summaries.update(
        summarise([_outcome(q, "naive", naive_run(client, q.question)) for q in questions])
    )
    summaries.update(
        summarise(
            [_outcome(q, "keyword", keyword_naive_run(client, q.question)) for q in questions]
        )
    )

    for name, forced in FIXED_PAIRINGS:
        fixed_outcomes: list[QuestionOutcome] = []
        planner = Planner(client, max_engines=2, max_rounds=1, force_engines=forced)
        for question in questions:
            try:
                result = planner.run(question.question, budget=budget)
            except FrugalError:
                continue
            fixed_outcomes.append(_outcome(question, name, result))
        summaries.update(summarise(fixed_outcomes))
        if verbose:
            s = summaries[name]
            print(f"  {name:<16} {s.answered}/{s.questions} answered, {s.searches} searches")

    for name, engines, rounds in ABLATIONS:
        if skip_deep and name in DEEP_ABLATIONS:
            continue
        outcomes: list[QuestionOutcome] = []
        planner = Planner(client, max_engines=engines, max_rounds=rounds)
        for question in questions:
            try:
                result = planner.run(question.question, budget=budget)
            except FrugalError:
                continue
            outcomes.append(_outcome(question, name, result))
        summaries.update(summarise(outcomes))
        if verbose:
            summary = summaries[name]
            print(
                f"  {name:<12} {summary.answered}/{summary.questions} answered, "
                f"{summary.searches} searches"
            )

    return summaries


def render_ablation(summaries: dict[str, StrategySummary]) -> str:
    """The recall-against-effort curve, in Markdown."""
    header = (
        "| configuration | answered | recall | searches | searches/question "
        "| answers/search |\n|---|---|---|---|---|---|"
    )
    rows = []
    order = ("naive", "keyword", *(f[0] for f in FIXED_PAIRINGS), *(a[0] for a in ABLATIONS))
    for name in order:
        summary = summaries.get(name)
        if summary is None:
            continue
        rows.append(
            f"| {name} | {summary.answered}/{summary.questions} | {summary.recall:.0%} "
            f"| {summary.searches} | {summary.searches_per_question:.1f} "
            f"| {summary.answers_per_search:.2f} |"
        )
    return "\n".join([header, *rows])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m frugal.benchmark",
        description="Compare planned search against the single-search baseline.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="serve only from recorded fixtures; spends nothing and fails on a miss",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what the run would cost, without issuing anything",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="sweep plan size, to show what each additional search buys",
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="omit the three-engine and two-round sweeps, which cost the most",
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="searches per question")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument("--cache", default=".frugal-cache", help="cache directory")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="where to write results JSON")
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.limit is not None:
        questions = questions[: args.limit]

    mode = CacheMode.REPLAY if args.replay else CacheMode.AUTO
    cache = ResponseCache(args.cache, mode=mode)

    if args.dry_run:
        return _report_dry_run(cache, questions, budget=args.budget)

    started = time.perf_counter()

    if args.ablate:
        with SerpApiClient(cache=cache) as client:
            summaries = run_ablation(
                client, questions, budget=args.budget, skip_deep=args.skip_deep
            )
            billed = client.log.searches_charged
        print("\n" + render_ablation(summaries))
        print(f"\nsearches billed this run: {billed}")
        Path(args.out).write_text(
            json.dumps(
                {"ablation": {k: v.as_dict() for k, v in summaries.items()}},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 0

    with SerpApiClient(cache=cache) as client:
        outcomes = run_benchmark(client, questions, budget=args.budget)
        billed = client.log.searches_charged

    summaries = summarise(outcomes)
    print("\n" + render_table(summaries))
    print(f"\nsearches billed this run: {billed}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")

    payload = {
        "questions": len(questions),
        "budget_per_question": args.budget,
        "searches_billed_this_run": billed,
        "summaries": {name: s.as_dict() for name, s in summaries.items()},
        "outcomes": [o.as_dict() for o in outcomes],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"results written to {args.out}")
    return 0


def _report_dry_run(cache: ResponseCache, questions: Sequence[Question], *, budget: int) -> int:
    """Project the cost of a full run without issuing anything."""
    client = SerpApiClient(cache=ResponseCache(cache.directory, mode=CacheMode.REPLAY))
    planner = Planner(client)

    total = 0
    print(f"{'question':<26} {'planned':>8} {'naive':>6} {'total':>6}")
    print("-" * 50)
    for question in questions:
        ceiling = planner.dry_run(question.question, budget=budget).ceiling
        line = ceiling + 1
        total += line
        print(f"{question.id:<26} {ceiling:>8} {1:>6} {line:>6}")

    print("-" * 50)
    print(f"{'ceiling for the full run':<26} {'':>8} {'':>6} {total:>6}")
    print(
        "\nThis is the worst case: every planned step issued, nothing saturating early "
        "and nothing served from cache.\nA real run costs less on both counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
