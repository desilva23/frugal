"""Turning a question into queries worth issuing.

Generating query variants is easy and mostly worthless. "monsoon onset India"
and "when does the monsoon start in India" return substantially the same page, so
issuing both costs two searches for one page of evidence. That is precisely the
waste this project exists to remove, and a reformulator that cheerfully produces
rephrasings makes it worse rather than better.

So the useful half of this module is the part that *refuses*. Candidates are
scored for novelty against the queries a plan has already issued, and anything
too close to one of them is dropped before it costs anything. A reformulation
earns its search by being substantively different — a different facet, a
different framing, or a shape a particular engine actually wants — not by being
differently worded.

Engine shaping is the clearest case. ``google_trends`` wants a bare term: asked
"is interest in electric vehicles growing over time?", the query that works is
"electric vehicles". Sending the whole sentence returns nothing useful, and no
amount of rephrasing fixes it.

Deterministic throughout, for the same reason the router is: the benchmark has
to reproduce without a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Words that carry no retrieval signal. Search engines discard these anyway, so
#: two queries differing only in stopwords are the same query and must not be
#: scored as different.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "have", "has", "had", "having",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
        "i", "me", "my", "we", "our", "you", "your", "it", "its", "they", "them",
        "this", "that", "these", "those", "there", "here",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
        "as", "into", "like", "through", "after", "over", "between", "out",
        "and", "or", "but", "if", "then", "than", "so", "because",
        "get", "got", "give", "tell", "find", "show", "know", "need", "want",
        "any", "some", "all", "each", "more", "most", "other", "such",
        "no", "not", "only", "own", "same", "very", "just", "now",
        "s", "t", "don", "please",
    }
)

#: Words that frame a question about a subject without being the subject.
#: "is interest in electric vehicles growing" is a question about electric
#: vehicles, and a term query built from "interest" retrieves the wrong thing.
_FRAMING_WORDS = frozenset(
    {
        "interest", "popularity", "demand", "trend", "trends", "trending",
        "growing", "growth", "rising", "falling", "declining", "decline",
        "compared", "versus", "vs", "difference", "best", "good", "better",
        "worth", "should", "people", "someone", "things", "stuff", "way", "ways",
    }
)

#: Words signalling the question wants current information. Kept out of the
#: keyword query — an engine sorting by date does not need to be told twice —
#: but used to decide whether a recency-narrowed variant is worth issuing.
_RECENCY_WORDS = frozenset(
    {"latest", "recent", "recently", "current", "today", "now", "new", "newest", "update"}
)

#: Above this token overlap, two queries will return substantially the same
#: results and the second one is not worth paying for. Chosen deliberately low:
#: the cost of dropping a marginally-different query is one missed page, while
#: the cost of issuing a redundant one is a search that buys nothing.
DEFAULT_MIN_NOVELTY = 0.25

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenise(text: str) -> tuple[str, ...]:
    """Reduce text to content tokens, in order, without duplicates."""
    words = _WORD.findall(text.lower())
    return tuple(dict.fromkeys(w for w in words if w not in _STOPWORDS and len(w) > 1))


def similarity(left: str, right: str) -> float:
    """Jaccard overlap of the content tokens of two queries.

    Chosen over an embedding for the reason everything here is deterministic:
    the benchmark must reproduce without a model. It is also inspectable, which
    matters when explaining why a query was dropped.

    The limitation is real and worth stating: this measures *lexical* overlap,
    so it catches near-duplicate wording but not paraphrase. "monsoon onset
    India" and "when does the monsoon start in India" score 0.5 here despite
    returning much the same page, because "onset" and "start" share no
    characters. Catching that would need embeddings, which would make the
    benchmark unrepeatable.

    The saturation monitor is the backstop. Reformulation drops redundancy that
    is cheap to detect before it costs anything; saturation catches the rest
    after one search has revealed it, and stops the plan. The two mechanisms
    cover different halves of the same problem, and the benchmark ablates them
    separately for exactly that reason.
    """
    a, b = set(tokenise(left)), set(tokenise(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novelty(candidate: str, issued: tuple[str, ...]) -> float:
    """How different ``candidate`` is from the most similar query already issued.

    ``1.0`` when nothing comparable has been issued, ``0.0`` when an identical
    query already has been.
    """
    if not issued:
        return 1.0
    return 1.0 - max(similarity(candidate, prior) for prior in issued)


@dataclass(frozen=True, slots=True)
class Reformulation:
    """One query a plan might issue, with why it was produced and kept."""

    query: str
    strategy: str
    engine: str
    novelty: float
    rationale: str

    def explain(self) -> str:
        return f'"{self.query}" [{self.strategy}, novelty {self.novelty:.2f}] — {self.rationale}'


def _keyword_query(question: str) -> str:
    """Content words only — what a search engine actually matches on."""
    return " ".join(tokenise(question))


def _core_terms(question: str, keep: int = 3) -> str:
    """The most salient terms, longest first.

    A crude salience proxy, but a deterministic one: longer content words are
    more often the domain terms that distinguish a question, and short ones more
    often the scaffolding around them.
    """
    tokens = [
        t for t in tokenise(question) if t not in _RECENCY_WORDS and t not in _FRAMING_WORDS
    ]
    if not tokens:
        # A question made entirely of framing has no core; fall back to content
        # words rather than returning nothing.
        tokens = list(tokenise(question))
    ranked = sorted(tokens, key=lambda t: (-len(t), tokens.index(t)))[:keep]
    return " ".join(sorted(ranked, key=tokens.index))


def _strip_recency(question: str) -> str:
    """Drop recency words, which narrow a query without adding retrieval signal."""
    return " ".join(t for t in tokenise(question) if t not in _RECENCY_WORDS)


#: Per-engine query shaping. Each entry is (strategy name, builder, rationale).
#: These are not rephrasings: each produces the shape its engine needs, and
#: sending the wrong shape wastes the search however well it is worded.
_ENGINE_SHAPES: dict[str, tuple[str, str]] = {
    "google_trends": (
        "bare-term",
        "trends matches a term, not a sentence; a full question returns nothing",
    ),
    "google_scholar": (
        "technical-terms",
        "scholar indexes paper text, so colloquial framing only dilutes the match",
    ),
    "google_shopping": (
        "product-terms",
        "shopping matches product names; question framing is noise",
    ),
    "google_jobs": (
        "role-terms",
        "jobs matches titles and descriptions, not questions",
    ),
    "google_maps": (
        "place-terms",
        "maps matches place names and categories",
    ),
    "google_patents": (
        "invention-terms",
        "patents indexes claim language, not questions",
    ),
}


def shape_for_engine(question: str, engine: str) -> tuple[str, str, str] | None:
    """Return ``(query, strategy, rationale)`` shaped for ``engine``, if it needs shaping."""
    entry = _ENGINE_SHAPES.get(engine)
    if entry is None:
        return None
    strategy, rationale = entry

    if engine == "google_trends":
        # Trends is the extreme case: it wants two or three words, not a sentence.
        return _core_terms(question, keep=3), strategy, rationale
    return _strip_recency(question), strategy, rationale


def reformulate(
    question: str,
    *,
    engine: str = "google",
    issued: tuple[str, ...] = (),
    limit: int = 3,
    min_novelty: float = DEFAULT_MIN_NOVELTY,
) -> list[Reformulation]:
    """Produce queries worth issuing against ``engine``, in priority order.

    Candidates below ``min_novelty`` against ``issued`` are dropped rather than
    returned, because issuing them would spend a search on a page the plan
    already has. The first call for an engine therefore returns more than later
    calls do, which is the intended shape: a plan should stop finding new angles
    before it stops finding budget.

    :raises ValueError: if ``question`` is blank.
    """
    if not question or not question.strip():
        raise ValueError("cannot reformulate an empty question")

    engine = engine.strip().lower()
    candidates: list[tuple[str, str, str]] = []

    shaped = shape_for_engine(question, engine)
    if shaped is not None:
        candidates.append(shaped)

    keyword = _keyword_query(question)
    if keyword:
        candidates.append(
            (keyword, "keyword", "content words only, which is what the index matches on")
        )

    candidates.append(
        (question.strip(), "verbatim", "the question as asked, for engines that parse phrasing")
    )

    core = _core_terms(question, keep=3)
    if core:
        candidates.append(
            (
                core,
                "core-terms",
                "broadened to the salient terms, to catch differently-worded pages",
            )
        )

    if any(word in tokenise(question) for word in _RECENCY_WORDS):
        stripped = _strip_recency(question)
        if stripped:
            candidates.append(
                (
                    stripped,
                    "recency-dropped",
                    "recency words narrow the match without adding signal",
                )
            )

    kept: list[Reformulation] = []
    seen_queries: list[str] = list(issued)

    for query, strategy, rationale in candidates:
        normalised = " ".join(query.split())
        if not normalised:
            continue

        score = novelty(normalised, tuple(seen_queries))
        if score < min_novelty:
            # Dropped before it costs anything. This is the mechanism, not an
            # optimisation around the edges of one.
            continue

        kept.append(
            Reformulation(
                query=normalised,
                strategy=strategy,
                engine=engine,
                novelty=round(score, 4),
                rationale=rationale,
            )
        )
        seen_queries.append(normalised)

        if len(kept) >= limit:
            break

    return kept
