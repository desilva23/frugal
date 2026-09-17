"""Deciding which engines can answer a question.

This is the core claim of the project: an agent that sends every question to web
search is leaving both accuracy and money on the table, because SerpApi exposes
engines that answer particular question shapes far better. A question about a
paper belongs in ``google_scholar``; a question about whether something is
gaining popularity belongs in ``google_trends``, which returns demand data no
list of links contains.

The router is deliberately **deterministic**. Three reasons, in order of
importance:

*The benchmark has to reproduce.* A model in this loop makes the published
result table unrepeatable, which would destroy the only artifact that
distinguishes this work from a demo.

*It costs nothing.* Routing runs before any search is issued, so a model call
here would be a per-question tax on a project whose entire thesis is cost.

*It is inspectable.* Every decision carries the signals that produced it, so a
wrong route can be diagnosed and fixed rather than re-prompted.

Scoring works by matching signals — lexical evidence that a question has a
particular shape — against engine profiles that declare which signals they
serve. Signals may also *suppress* engines: "patent" is strong evidence for
``google_patents`` and strong evidence against ``google_shopping``, and encoding
the negative case is what stops a multi-engine plan wasting searches on
plausible-looking wrong sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Weight given to web search regardless of signals. It answers most questions
#: adequately, so it is never routed away from entirely — it is the floor a
#: specialised engine has to beat, not a competitor it has to displace.
_WEB_SEARCH_PRIOR = 1.0

#: Minimum score for an engine to be worth issuing a search against. Below this,
#: a match is one incidental word rather than a question shape.
_SELECTION_FLOOR = 0.35


#: Words that invert the meaning of a signal appearing shortly after them.
#: "recent news" argues for google_news; "not recent news" argues against it, and
#: a bag-of-words matcher reads both the same way.
_NEGATORS = frozenset(
    {
        "not", "no", "without", "dont", "doesnt", "didnt", "isnt", "arent",
        "exclude", "excluding", "except", "never", "avoid", "ignore", "besides",
        "other", "rather", "unrelated", "nothing",
    }
)

#: How many words before a match are searched for a negator. Wide enough to
#: cover "do not show me recent news", narrow enough that a negation earlier in
#: an unrelated clause does not reach across and cancel it.
_NEGATION_WINDOW = 5

_WORDS = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _is_negated(question: str, start: int) -> bool:
    """Whether a match at ``start`` sits shortly after a negation."""
    preceding: list[str] = _WORDS.findall(question[:start].lower())
    return any(
        word.replace("'", "") in _NEGATORS for word in preceding[-_NEGATION_WINDOW:]
    )


def _inside_proper_noun(question: str, match: re.Match[str]) -> bool:
    """Whether a match is part of a name rather than a topic.

    "Steve Jobs biography" is not an employment question, but ``\bjobs\b``
    matches it and scores it above every other engine. The test is capitalisation
    in context: the matched word is capitalised, it is not the first word of the
    question, and the word before it is also capitalised. That catches personal
    and company names generically rather than by keeping a list of them.

    A sentence-initial capital is ignored, so "Jobs in Chennai" still routes to
    google_jobs.
    """
    text = match.group(0)
    if not text[:1].isupper():
        return False

    before: list[str] = _WORDS.findall(question[: match.start()])
    if not before:
        # Sentence-initial: capitalised because it starts the question.
        return False
    return bool(before[-1][:1].isupper())


@dataclass(frozen=True, slots=True)
class Signal:
    """Lexical evidence that a question has a particular shape.

    Patterns are matched case-insensitively against the whole question with word
    boundaries, so ``"job"`` does not fire on ``"jobless"`` and, more to the
    point, ``"price"`` does not fire on ``"priceless"``.
    """

    name: str
    patterns: tuple[str, ...]
    weight: float = 1.0

    def matches(self, question: str) -> tuple[str, ...]:
        """Return the words that fired, in the question's own wording.

        The matched text is returned rather than the pattern so that a plan
        trace reads "local (near me, open now)" rather than showing a regex to
        someone trying to understand why an engine was chosen.

        Two kinds of match are discarded: one sitting inside a proper noun, and
        one shortly after a negation. Both are cases where the word is present
        and the meaning is absent, which is the characteristic failure of
        matching a bag of words.
        """
        found: list[str] = []
        for pattern in self.patterns:
            for match in re.finditer(rf"\b{pattern}\b", question, re.IGNORECASE):
                if _is_negated(question, match.start()):
                    continue
                if _inside_proper_noun(question, match):
                    continue
                found.append(match.group(0).lower())
                break
        return tuple(dict.fromkeys(found))


SIGNALS: dict[str, Signal] = {
    "recency": Signal(
        "recency",
        (
            "latest", "recent", "recently", "news", "today", "yesterday", "this week",
            "this month", "breaking", "just announced", "announced", "update", "updates",
            "current", "right now", "so far this year",
        ),
        weight=1.2,
    ),
    "scholarly": Signal(
        "scholarly",
        (
            "paper", "papers", "study", "studies", "research", "journal", "cited",
            "citation", "citations", "peer.reviewed", "literature", "findings",
            "publication", "preprint", "arxiv", "methodology", "authors",
        ),
        weight=1.5,
    ),
    "commerce": Signal(
        "commerce",
        (
            "price", "prices", "pricing", "cost", "costs", "buy", "cheapest",
            "cheaper", "deal", "deals", "discount", "offer", "offers", "under",
            "budget", "brand", "brands", "model", "sale", "purchase", "worth buying",
        ),
        weight=1.3,
    ),
    "local": Signal(
        "local",
        (
            "near me", "nearby", "near", "open now", "opening hours", "hours",
            # "address" is genuinely local in "the address of X" but not in
            # "IP address" or "email address", which would otherwise misroute.
            r"(?<!ip )(?<!email )(?<!mac )(?<!web )address",
            "directions", "restaurant", "restaurants", "cafe", "cafes",
            "shop", "shops", "store", "stores", "clinic", "hospital", "closest",
            "in my area", "around here",
        ),
        weight=1.4,
    ),
    "employment": Signal(
        "employment",
        (
            "job", "jobs", "hiring", "hire", "salary", "salaries", "vacancy",
            "vacancies", "career", "careers", "recruit", "recruiting", "recruitment",
            "openings", "role", "roles", "position", "positions", "employer",
            "employment", "apply", "applicants", "staffing", "headcount",
            # "work" on its own fires on "how does photosynthesis work", so only
            # the phrasings that unambiguously mean employment are listed.
            "work as", "work at", "work for", "looking for work", "find work",
            "get a job", "land a job",
        ),
        weight=1.5,
    ),
    "patent": Signal(
        "patent",
        (
            "patent", "patents", "patented", "invention", "inventor", "assignee",
            "prior art", "filed", "filing", "intellectual property",
        ),
        weight=1.7,
    ),
    "trend": Signal(
        "trend",
        (
            "trend", "trends", "trending", "growing", "growth", "declining",
            "decline", "over time", "popularity", "interest", "demand",
            "rising", "falling", "since", "compared to last",
        ),
        weight=1.4,
    ),
}


@dataclass(frozen=True, slots=True)
class EngineProfile:
    """What an engine is good for, and what calling it costs.

    :param serves: Signals this engine answers well, mapped to a multiplier.
    :param suppressed_by: Signals that are evidence *against* this engine.
        Encoding the negative case is what stops a plan spending searches on a
        plausible-looking wrong source.
    :param prior: Score before any signal fires.
    :param cost: Searches consumed per call, so the budget governor can compare
        engines that are not equally priced.
    """

    engine: str
    serves: dict[str, float] = field(default_factory=dict)
    suppressed_by: dict[str, float] = field(default_factory=dict)
    prior: float = 0.0
    cost: int = 1


PROFILES: tuple[EngineProfile, ...] = (
    EngineProfile(
        engine="google",
        # Web search answers most things adequately. It is the floor a
        # specialised engine has to beat.
        serves={"recency": 0.3, "commerce": 0.2, "scholarly": 0.2},
        prior=_WEB_SEARCH_PRIOR,
    ),
    EngineProfile(
        engine="google_news",
        serves={"recency": 1.6, "trend": 0.4},
        # A question about papers or patents wanting "recent" ones is still not
        # a news question.
        suppressed_by={"scholarly": 0.5, "patent": 0.6, "local": 1.1},
    ),
    EngineProfile(
        engine="google_scholar",
        serves={"scholarly": 1.8},
        suppressed_by={"commerce": 0.8, "local": 0.8, "employment": 0.5},
    ),
    EngineProfile(
        engine="google_shopping",
        serves={"commerce": 1.7},
        suppressed_by={"scholarly": 0.9, "patent": 0.9, "employment": 0.6},
    ),
    EngineProfile(
        engine="google_jobs",
        serves={"employment": 1.9},
        suppressed_by={"commerce": 0.4, "scholarly": 0.4},
    ),
    EngineProfile(
        engine="google_maps",
        serves={"local": 1.8},
        suppressed_by={"scholarly": 0.9, "patent": 0.9, "trend": 0.4},
    ),
    EngineProfile(
        engine="google_patents",
        serves={"patent": 2.0, "scholarly": 0.4},
        suppressed_by={"commerce": 0.7, "local": 0.9, "recency": 0.3},
    ),
    EngineProfile(
        engine="google_trends",
        serves={"trend": 1.9},
        suppressed_by={"local": 0.7, "scholarly": 0.4},
    ),
)

_PROFILES_BY_ENGINE = {profile.engine: profile for profile in PROFILES}


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """One engine's suitability for a question, with the evidence that produced it."""

    engine: str
    score: float
    cost: int
    reasons: tuple[str, ...]

    def explain(self) -> str:
        """A single line for the plan trace and the demo."""
        because = "; ".join(self.reasons) if self.reasons else "general-purpose fallback"
        return f"{self.engine} (score {self.score:.2f}) — {because}"


def detect_signals(question: str) -> dict[str, tuple[str, ...]]:
    """Return every signal that fired, mapped to the words that fired it."""
    return {
        name: matched
        for name, signal in SIGNALS.items()
        if (matched := signal.matches(question))
    }


def route(
    question: str,
    *,
    limit: int | None = None,
    floor: float = _SELECTION_FLOOR,
) -> list[RoutingDecision]:
    """Rank engines by their fitness for ``question``.

    Returns decisions above ``floor``, highest first, capped at ``limit``. Web
    search is always included even when nothing fires: a question with no
    recognisable shape is a question for general search, and returning an empty
    plan would be worse than returning an unspecialised one.
    """
    if not question or not question.strip():
        raise ValueError("cannot route an empty question")

    fired = detect_signals(question)
    decisions: list[RoutingDecision] = []

    for profile in PROFILES:
        score = profile.prior
        reasons: list[str] = []

        for signal_name, multiplier in profile.serves.items():
            if signal_name not in fired:
                continue
            contribution = SIGNALS[signal_name].weight * multiplier
            score += contribution
            words = ", ".join(fired[signal_name][:3])
            reasons.append(f"{signal_name} ({words})")

        for signal_name, penalty in profile.suppressed_by.items():
            if signal_name not in fired:
                continue
            score -= SIGNALS[signal_name].weight * penalty
            reasons.append(f"-{signal_name}")

        if score >= floor:
            decisions.append(
                RoutingDecision(
                    engine=profile.engine,
                    score=round(score, 4),
                    cost=profile.cost,
                    reasons=tuple(reasons),
                )
            )

    # Ties broken by the profile order above, which puts general search first —
    # a deterministic ordering matters because the benchmark must reproduce.
    decisions.sort(key=lambda d: -d.score)

    if not any(d.engine == "google" for d in decisions):
        decisions.append(
            RoutingDecision(engine="google", score=_WEB_SEARCH_PRIOR, cost=1, reasons=())
        )
        decisions.sort(key=lambda d: -d.score)

    return decisions[:limit] if limit is not None else decisions


def profile_for(engine: str) -> EngineProfile | None:
    """Look up an engine's profile, or ``None`` if it has none."""
    return _PROFILES_BY_ENGINE.get(engine.strip().lower())
