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
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from frugal.cache import CacheMode, ResponseCache, compute_key
from frugal.client import SerpApiClient
from frugal.errors import FrugalError
from frugal.planner import (
    DEFAULT_BUDGET,
    Planner,
    PlanResult,
    PlanStep,
    baseline_step,
    keyword_naive_run,
    naive_run,
    parameterised_naive_run,
)
from frugal.schema import Evidence, Series

#: Markers are matched on word boundaries, not as bare substrings.
#:
#: Substring containment was the original implementation and it made several
#: questions unfalsifiable. "ai" matched "said" and "available"; "upi" matched
#: "occupied"; "rs" matched "years" and "offers"; "100" matched "21000". Eight of
#: the thirty questions carry a marker of three characters or fewer, so those
#: questions scored for every strategy regardless of what it retrieved, and every
#: recall figure was inflated by an unknown amount.
#:
#: A boundary is only asserted where the marker's own edge is alphanumeric, so a
#: marker that begins or ends in punctuation -- the rupee sign, "°c" -- still
#: matches. re.escape keeps a marker containing regex metacharacters literal.
#:
#: A marker that is a single plain word also matches its plural. Word boundaries
#: fixed substring matching and introduced the opposite fault: "hospital" stopped
#: matching "hospitals" and "patent" stopped matching "patents", so a result
#: reading "major hospitals in Coimbatore" failed a question whose marker was
#: "hospital" -- and every patent question's "patent" marker failed on the plural
#: that patent records mostly use. This is the rule the router adopted for the
#: same fault. Multi-word markers are matched as written, so a phrase that has a
#: plural lists it.
#:
#: Short markers are excluded, because pluralising one does not produce a longer
#: form of the same word -- it produces a different short token that collides
#: with real ones. "rs" plus a plural matched "RSS", which appears on any page
#: with a feed link, and eight commerce questions carry "rs" as the alternative
#: that stands for a rupee price; "gil" matched the name "Giles". Both would
#: have been satisfied by pages containing no answer at all, which is the fault
#: word boundaries were introduced to remove, arriving by a second route.
#:
#: So a marker must be long enough that its plural is still distinctive. A
#: question that genuinely needs a short plural lists it: "llms" alongside
#: "llm". Stating it is better than inferring it, since the inference is what
#: keeps going wrong.
_MARKER_CACHE: dict[str, re.Pattern[str]] = {}
_PLAIN_MARKER = re.compile(r"^[a-z][a-z\-]*$", re.IGNORECASE)

#: Below this length, a pluralised marker collides with unrelated words more
#: readily than it catches a real plural.
_MIN_PLURAL_LENGTH = 4


def marker_pattern(marker: str) -> re.Pattern[str]:
    """Compile a marker into a boundary-respecting pattern, memoised."""
    cached = _MARKER_CACHE.get(marker)
    if cached is None:
        body = re.escape(marker)
        if len(marker) >= _MIN_PLURAL_LENGTH and _PLAIN_MARKER.match(marker):
            body += r"(?:e?s)?"
        prefix = r"\b" if marker[:1].isalnum() else ""
        suffix = r"\b" if marker[-1:].isalnum() else ""
        cached = re.compile(prefix + body + suffix, re.IGNORECASE)
        _MARKER_CACHE[marker] = cached
    return cached


def marker_matches(marker: str, text: str) -> bool:
    """Whether ``marker`` occurs in ``text`` as a word rather than a fragment."""
    return marker_pattern(marker).search(text) is not None


QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "results.json"

#: The sweep writes a different shape from the main run -- summaries per plan
#: size, no per-question outcomes -- so it gets its own file. Sharing a default
#: meant an --ablate run silently replaced the headline table with a payload
#: that did not contain it.
ABLATION_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "ablation.json"

#: Four strategies, each adding one mechanism to the one before it, so that
#: every gap isolates a single thing:
#:
#:   naive           the question verbatim to web search -- what many agents do
#:   keyword         + reformulation
#:   parameterised   + the engine parameters gl, hl, location, geo, as_ylo
#:   planned         + routing across engines, under a budget
#:
#: The third was added after a review pointed out that the keyword-to-planned gap
#: contained both parameters and routing while the README credited all of it to
#: routing. Two strategies could not separate reformulation from routing; three
#: could not separate routing from parameters.
STRATEGIES = ("naive", "keyword", "parameterised", "planned")


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
    #: Whether the evidence included a time series. Descriptive only: it records
    #: which kind of evidence came back, and is deliberately not scored.
    #:
    #: There used to be an `answered_in_the_right_modality` property here,
    #: reported as a figure out of thirty and quoted in the README as "the one
    #: consistent difference". It was not a measurement. Only google_trends
    #: produces a Series and only the routed strategy calls it, so the baselines
    #: scored exactly thirty minus the four questions marked expects_series, and
    #: the routed strategy scored thirty. The number described which engines a
    #: configuration calls, which is already known before running anything.
    #:
    #: This is the same fault that was removed from recall earlier -- a metric
    #: defined by the thing it measures -- so the fix is to stop scoring it
    #: rather than to relocate it again. Which strategies reached a series is
    #: still worth stating, and the README states it as a fact about the engines
    #: rather than as a result.
    structured: bool = False
    #: Whether the question is one a series answers better than prose.
    wanted_structured: bool = False

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
            if index not in satisfied and any(marker_matches(alt, text) for alt in group):
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


#: Step errors that mean the fixture set is incomplete, rather than that a
#: search failed. Matched against the ``Name: message`` form a step records.
_REPLAY_GAPS = ("CacheMiss:", "CacheCorrupt:")


def _reject_replay_gaps(question: Question, strategy: str, result: PlanResult) -> None:
    """Refuse to score a replayed question whose fixtures are incomplete.

    The planner treats a failed step as survivable, and during a live run that
    is right: one engine being unreachable should not throw away a plan already
    paid for. Under replay it is not survivable. A missing fixture means the
    code now plans a step that the recorded run never took, and the benchmark
    was printing a table anyway -- counting the step it could not replay in the
    searches column, and scoring the question on the evidence of the steps that
    happened to survive.

    Two steps were in exactly this state when this check was written, because
    routing and reformulation both changed after the fixtures were exported. The
    published table did not announce it. A result nobody can reproduce is worth
    less than no result, so this stops the run instead.
    """
    for step in result.steps:
        if step.error and step.error.startswith(_REPLAY_GAPS):
            raise FrugalError(
                f"replay is missing a fixture for {question.id!r} ({strategy}): "
                f"{step.step.engine} {step.step.query!r} -- {step.error}. "
                f"The fixtures no longer cover what this code plans; re-record "
                f"them with a fresh --cache directory before publishing a table."
            )


def _outcome(
    question: Question,
    strategy: str,
    result: PlanResult,
    *,
    replay: bool = False,
) -> QuestionOutcome:
    if replay:
        _reject_replay_gaps(question, strategy, result)
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
    replay = client.cache.mode is CacheMode.REPLAY
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
                elif strategy == "parameterised":
                    result = parameterised_naive_run(client, question.question)
                else:
                    result = planner.run(question.question, budget=budget)
            except FrugalError as exc:
                # One strategy failing on one question must not lose the rest of
                # the run, which may have cost real searches to get this far.
                if verbose:
                    print(f"    {strategy:<8} FAILED  {type(exc).__name__}: {exc}")
                continue

            outcome = _outcome(question, strategy, result, replay=replay)
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
DEEP_ABLATIONS = frozenset({"routed-3x1"})


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
    replay = client.cache.mode is CacheMode.REPLAY

    summaries.update(
        summarise(
            [
                _outcome(q, "naive", naive_run(client, q.question), replay=replay)
                for q in questions
            ]
        )
    )
    summaries.update(
        summarise(
            [
                _outcome(q, "keyword", keyword_naive_run(client, q.question), replay=replay)
                for q in questions
            ]
        )
    )
    summaries.update(
        summarise(
            [
                _outcome(
                    q,
                    "parameterised",
                    parameterised_naive_run(client, q.question),
                    replay=replay,
                )
                for q in questions
            ]
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
            fixed_outcomes.append(_outcome(question, name, result, replay=replay))
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
            outcomes.append(_outcome(question, name, result, replay=replay))
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
    order = (
        "naive",
        "keyword",
        "parameterised",
        *(f[0] for f in FIXED_PAIRINGS),
        *(a[0] for a in ABLATIONS),
    )
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


def _report_spend(log: Any) -> None:
    """Print what the run recorded, and what it may have paid for and lost.

    Reported even when it is zero, because the useful case is noticing that it
    is not. A gap here means SerpApi ran searches whose responses never reached
    the cache, and those are billed exactly like the ones that did.
    """
    print(f"\nsearches billed this run: {log.searches_charged}")
    if log.unrecorded_attempts:
        print(
            f"requests that produced no recorded search: {log.unrecorded_attempts} "
            f"(of {log.attempts} sent) -- SerpApi may have billed these"
        )


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
    parser.add_argument(
        "--questions",
        default=None,
        help="question set to run (default: benchmarks/questions.json)",
    )
    parser.add_argument("--cache", default=".frugal-cache", help="cache directory")
    parser.add_argument(
        "--out",
        default=None,
        help="where to write results JSON (default: benchmarks/results.json, "
        "or benchmarks/ablation.json with --ablate)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(ABLATION_PATH if args.ablate else RESULTS_PATH)

    questions = load_questions(
        Path(args.questions) if args.questions else QUESTIONS_PATH
    )
    if args.limit is not None:
        questions = questions[: args.limit]

    mode = CacheMode.REPLAY if args.replay else CacheMode.AUTO
    # A benchmark compares strategies, so every strategy has to see one snapshot
    # of the web. Freshness windows break that. The working cache expires web
    # results after six hours and news after fifteen minutes but keeps scholarly
    # results for a week, so a live run two days after recording re-fetched some
    # engines and reused others -- mixing two snapshots in a single table. Two
    # days of drift moved the baseline's recall by four questions, twice the
    # effect being measured, and cost seventy-three searches to learn nothing.
    #
    # So the benchmark never treats a recorded entry as stale. A step already
    # recorded is reused as it was; only a step never recorded is fetched. To
    # take a genuinely new snapshot, point --cache at an empty directory, so that
    # every strategy is recorded in the same run.
    cache = ResponseCache(args.cache, mode=mode, ttls={}, fallback_ttl=float("inf"))

    if args.dry_run:
        return _report_dry_run(
            cache,
            questions,
            budget=args.budget,
            ablate=args.ablate,
            skip_deep=args.skip_deep,
        )

    started = time.perf_counter()

    if args.ablate:
        with SerpApiClient(cache=cache) as client:
            summaries = run_ablation(
                client, questions, budget=args.budget, skip_deep=args.skip_deep
            )
            billed = client.log.searches_charged
            spend = client.log
        print("\n" + render_ablation(summaries))
        _report_spend(spend)
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
        spend = client.log

    summaries = summarise(outcomes)
    print("\n" + render_table(summaries))
    _report_spend(spend)
    print(f"elapsed: {time.perf_counter() - started:.1f}s")

    payload = {
        "questions": len(questions),
        "budget_per_question": args.budget,
        "searches_billed_this_run": billed,
        "spend": spend.as_dict(),
        "summaries": {name: s.as_dict() for name, s in summaries.items()},
        "outcomes": [o.as_dict() for o in outcomes],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"results written to {args.out}")
    return 0


def _projected_arms(*, ablate: bool, skip_deep: bool) -> list[str]:
    """The arms a run would execute, in the order it would execute them.

    Order matters to the projection, because the second arm to issue an
    identical search does not pay for it.
    """
    if not ablate:
        return list(STRATEGIES)
    return [
        "naive",
        "keyword",
        "parameterised",
        *(name for name, _ in FIXED_PAIRINGS),
        *(
            name
            for name, _, _ in ABLATIONS
            if not (skip_deep and name in DEEP_ABLATIONS)
        ),
    ]


def _steps_for_arm(
    arm: str,
    question: str,
    *,
    planners: dict[str, Planner],
    budget: int,
) -> list[PlanStep]:
    """Every search one arm would issue for one question, without issuing it."""
    if arm in STRATEGIES[:3]:
        return [baseline_step(arm, question)]
    return list(planners[arm].dry_run(question, budget=budget).steps)


def _report_dry_run(
    cache: ResponseCache,
    questions: Sequence[Question],
    *,
    budget: int,
    ablate: bool = False,
    skip_deep: bool = False,
) -> int:
    """Project what a run would cost, without issuing anything.

    This used to project two arms of a four-arm benchmark and present the total
    as the cost of the run. Asked about a hundred-question snapshot it answered
    266 against an actual 636 -- under half. In a project whose whole subject is
    knowing the price before paying it, a preview that cheap-sells the bill by
    more than half is the wrong thing to ship.

    So the projection now enumerates the exact searches every arm would issue,
    from the same definitions the run uses, and prices them the way the client
    prices them: a search already recorded costs nothing, and two arms issuing
    an identical search pay once between them. What remains approximate is
    stated in the output rather than folded silently into a number.
    """
    client = SerpApiClient(cache=ResponseCache(cache.directory, mode=CacheMode.REPLAY))

    planners: dict[str, Planner] = {"planned": Planner(client)}
    for name, forced in FIXED_PAIRINGS:
        planners[name] = Planner(client, max_engines=2, max_rounds=1, force_engines=forced)
    for name, engines, rounds in ABLATIONS:
        planners[name] = Planner(client, max_engines=engines, max_rounds=rounds)

    arms = _projected_arms(ablate=ablate, skip_deep=skip_deep)
    seen: set[str] = set()
    rows: list[tuple[str, int, int, int]] = []
    multi_round = False

    for arm in arms:
        issued = free = billed = 0
        for question in questions:
            for step in _steps_for_arm(
                arm, question.question, planners=planners, budget=budget
            ):
                multi_round = multi_round or step.round_number > 1
                issued += 1
                params = {k: v for k, v in step.params.items() if v is not None}
                key = compute_key(step.engine, params)
                if key in seen or cache.peek(step.engine, params) is not None:
                    seen.add(key)
                    free += 1
                    continue
                seen.add(key)
                billed += 1
        rows.append((arm, issued, free, billed))

    width = max(len("total"), *(len(name) for name, *_ in rows))
    print(f"{'arm':<{width}}  {'searches':>8}  {'free':>6}  {'billed':>6}")
    print("-" * (width + 26))
    for name, issued, free, billed in rows:
        print(f"{name:<{width}}  {issued:>8}  {free:>6}  {billed:>6}")
    print("-" * (width + 26))
    total_issued = sum(row[1] for row in rows)
    total_free = sum(row[2] for row in rows)
    total_billed = sum(row[3] for row in rows)
    print(f"{'total':<{width}}  {total_issued:>8}  {total_free:>6}  {total_billed:>6}")

    print(
        f"\n{len(questions)} questions, budget {budget} per question, "
        f"cache {cache.directory}."
    )
    print(
        "Free means already recorded, or issued by an earlier arm in this same run.\n"
        "Billed is the worst case: every planned step issued, nothing saturating\n"
        "early. A plan that saturates stops sooner, so a real run bills this or less."
    )
    if multi_round:
        print(
            "\nRounds after the first adapt to what earlier rounds retrieved, so their\n"
            "queries are not knowable until the run happens. They are priced here as\n"
            "planned, which is their ceiling, and they may in fact collide with a\n"
            "search already paid for."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
