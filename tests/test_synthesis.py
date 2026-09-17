"""Tests for answer synthesis.

Synthesis sits outside everything the benchmark measures, so the bar here is not
answer quality -- that is the model's business -- but that the answer stays tied
to the evidence. A citation the model invented must not reach the reader as a
citation they cannot follow, and an empty evidence set must never reach the model
at all, because the only answer it could give would come from its own memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from frugal.errors import FrugalError, TransportError
from frugal.schema import Document, Provenance, Series, SeriesPoint
from frugal.synthesis import (
    _BRACKET_ALIASES,
    MAX_EVIDENCE,
    Answer,
    Synthesiser,
    SynthesisUnavailable,
    extract_citations,
    normalise_brackets,
    render_evidence,
)

NOW = datetime(2026, 9, 17, tzinfo=UTC)


def doc(title: str, **kwargs: object) -> Document:
    return Document(title=title, provenance=Provenance("google", "q", 1, NOW), **kwargs)  # type: ignore[arg-type]


def series(name: str = "term") -> Series:
    points = tuple(
        SeriesPoint(timestamp=datetime(2026, 1, i + 1, tzinfo=UTC), value=float(i * 10))
        for i in range(1, 4)
    )
    return Series(name=name, points=points, provenance=Provenance("google_trends", "q", 1, NOW))


def replying(content: str) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def synthesiser(transport: httpx.MockTransport) -> Synthesiser:
    return Synthesiser(api_key="test-key", http=httpx.Client(transport=transport))


# --------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------


def test_evidence_is_numbered_for_citation() -> None:
    rendered = render_evidence([doc("first"), doc("second")])
    assert "[1] first" in rendered
    assert "[2] second" in rendered


def test_a_series_is_rendered_as_its_summary_not_its_points() -> None:
    """Fifty timestamped numbers would crowd out every other result."""
    rendered = render_evidence([series("electric vehicles")])
    assert "TIME SERIES" in rendered
    assert "observations" in rendered


def test_long_snippets_are_truncated_rather_than_dropped() -> None:
    rendered = render_evidence([doc("t", snippet="x" * 1000)])
    assert "…" in rendered
    assert len(rendered) < 1000


def test_only_a_bounded_number_of_items_is_offered() -> None:
    rendered = render_evidence([doc(f"d{i}") for i in range(100)])
    assert f"[{MAX_EVIDENCE}]" in rendered
    assert f"[{MAX_EVIDENCE + 1}]" not in rendered


def test_url_and_source_reach_the_prompt() -> None:
    rendered = render_evidence([doc("t", url="https://example.com/a", source="Reuters")])
    assert "https://example.com/a" in rendered
    assert "Reuters" in rendered


# --------------------------------------------------------------------------
# Citations must be real
# --------------------------------------------------------------------------


def test_valid_citations_are_resolved() -> None:
    text, citations = extract_citations("The answer is x [1].", [doc("source one")])
    assert len(citations) == 1
    assert citations[0].title == "source one"
    assert "[1]" in text


def test_invented_citations_are_stripped_from_the_text() -> None:
    """A number referring to nothing must not reach the reader as a citation."""
    text, citations = extract_citations("Claim [1] and another [9].", [doc("only one")])
    assert citations and len(citations) == 1
    assert "[9]" not in text


def test_a_repeated_citation_is_listed_once() -> None:
    _, citations = extract_citations("x [1] y [1] z [1]", [doc("one")])
    assert len(citations) == 1


def test_citations_are_listed_in_order() -> None:
    evidence = [doc("a"), doc("b"), doc("c")]
    _, citations = extract_citations("[3] then [1] then [2]", evidence)
    assert [c.number for c in citations] == [1, 2, 3]


def test_zero_is_not_a_valid_citation() -> None:
    _, citations = extract_citations("see [0]", [doc("a")])
    assert citations == ()


def test_a_series_citation_carries_its_name() -> None:
    _, citations = extract_citations("[1]", [series("electric vehicles")])
    assert citations[0].title == "electric vehicles"
    assert citations[0].engine == "google_trends"


def test_an_uncited_answer_is_reported_as_ungrounded() -> None:
    answer = Answer(text="I think so.", citations=(), model="m", evidence_offered=3, elapsed_ms=1)
    assert not answer.grounded


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------


def test_an_answer_is_returned_with_its_citations() -> None:
    result = synthesiser(replying("Interest is rising [1].")).answer("q?", [doc("a source")])
    assert "rising" in result.text
    assert result.grounded


def test_no_evidence_never_reaches_the_model() -> None:
    """With nothing to read, any answer would come from the model's memory."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "invented"}}]})

    result = synthesiser(httpx.MockTransport(handler)).answer("q?", [])
    assert calls == []
    assert not result.grounded
    assert "nothing to answer from" in result.text


def test_the_question_and_evidence_both_reach_the_prompt() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok [1]"}}]})

    synthesiser(httpx.MockTransport(handler)).answer("why is the sky blue?", [doc("Rayleigh")])
    prompt = str(sent[0])
    assert "why is the sky blue?" in prompt
    assert "Rayleigh" in prompt


# --------------------------------------------------------------------------
# Failure modes, each with its own remedy
# --------------------------------------------------------------------------


def test_a_rejected_key_says_which_key() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(SynthesisUnavailable, match="GROQ_API_KEY"):
        synthesiser(transport).answer("q?", [doc("a")])


def test_an_unknown_model_says_model_names_change() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(404, json={"error": "no model"}))
    with pytest.raises(SynthesisUnavailable, match="not found"):
        synthesiser(transport).answer("q?", [doc("a")])


def test_a_server_error_is_a_transport_error_not_a_configuration_one() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(500, text="boom"))
    with pytest.raises(TransportError):
        synthesiser(transport).answer("q?", [doc("a")])


def test_an_unreachable_endpoint_is_reported_clearly() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(TransportError, match="could not reach"):
        synthesiser(httpx.MockTransport(handler)).answer("q?", [doc("a")])


def test_a_malformed_response_is_rejected() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(TransportError, match="unexpected response"):
        synthesiser(transport).answer("q?", [doc("a")])


def test_an_empty_answer_is_rejected() -> None:
    with pytest.raises(TransportError, match="empty answer"):
        synthesiser(replying("   ")).answer("q?", [doc("a")])


def test_a_missing_key_explains_how_to_set_one_and_that_it_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> str:
        raise FrugalError("no key")

    monkeypatch.setattr("frugal.synthesis.resolve_key", refuse)
    with pytest.raises(SynthesisUnavailable) as excinfo:
        Synthesiser()
    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert "works without it" in message


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_an_injected_client_is_not_closed() -> None:
    http = httpx.Client(transport=replying("ok"))
    with Synthesiser(api_key="k", http=http):
        pass
    assert not http.is_closed


# --------------------------------------------------------------------------
# Bracket variants
# --------------------------------------------------------------------------
#
# Characters are built from code points rather than written literally: a file
# full of look-alike brackets is exactly the confusion under test, and a reader
# should not have to guess which one they are looking at.

CJK_OPEN, CJK_CLOSE = chr(0x3010), chr(0x3011)


def test_cjk_brackets_are_recognised_as_citations() -> None:
    """Observed from gpt-oss-120b: a cited answer was reported as ungrounded.

    Calling a correctly cited answer ungrounded is a false accusation of
    hallucination, which is worse than failing to notice a citation.
    """
    text, citations = extract_citations(f"Interest is falling{CJK_OPEN}1{CJK_CLOSE}.", [doc("s")])
    assert len(citations) == 1
    assert "[1]" in text


def test_every_declared_bracket_variant_resolves() -> None:
    """Derived from the module, so a newly added alias is covered by this test."""
    openers = [char for char, ascii_char in _BRACKET_ALIASES.items() if ascii_char == "["]
    closers = [char for char, ascii_char in _BRACKET_ALIASES.items() if ascii_char == "]"]
    assert openers and len(openers) == len(closers)

    for opener, closer in zip(openers, closers, strict=True):
        _, citations = extract_citations(f"claim {opener}1{closer}", [doc("a")])
        assert len(citations) == 1, f"{opener!r}{closer!r} was not recognised"


def test_normalisation_leaves_ascii_untouched() -> None:
    text, citations = extract_citations("claim [1]", [doc("a")])
    assert text == "claim [1]"
    assert len(citations) == 1


def test_an_invalid_citation_in_wide_brackets_is_still_stripped() -> None:
    text, _ = extract_citations(f"claim {CJK_OPEN}9{CJK_CLOSE}", [doc("only one")])
    assert "9" not in text


def test_normalise_brackets_is_idempotent() -> None:
    once = normalise_brackets(f"{CJK_OPEN}1{CJK_CLOSE}")
    assert normalise_brackets(once) == once == "[1]"


# --------------------------------------------------------------------------
# Transient failures
# --------------------------------------------------------------------------
#
# The retrieval is already paid for by the time synthesis runs, so a rate limit
# that clears in a second should not cost the write-up. Free tiers allow few
# requests per minute, and a demo asking several questions in a row will meet one.


def retrying(*statuses: int, then: str = "answered [1]") -> httpx.MockTransport:
    """Fail with each status in turn, then succeed."""
    remaining = list(statuses)

    def handler(_request: httpx.Request) -> httpx.Response:
        if remaining:
            return httpx.Response(remaining.pop(0), text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": then}}]})

    return httpx.MockTransport(handler)


def patient(transport: httpx.MockTransport, **kwargs: object) -> Synthesiser:
    return Synthesiser(
        api_key="test-key",
        http=httpx.Client(transport=transport),
        sleep=lambda _seconds: None,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_rate_limit_is_retried_rather_than_abandoned() -> None:
    result = patient(retrying(429)).answer("q?", [doc("a source")])
    assert result.grounded


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_every_transient_status_is_retried(status: int) -> None:
    assert patient(retrying(status)).answer("q?", [doc("a")]).text


def test_a_persistent_rate_limit_eventually_gives_up() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(429, text="slow down"))
    with pytest.raises(TransportError, match="after 3 attempts"):
        patient(transport).answer("q?", [doc("a")])


def test_retries_are_bounded_by_max_attempts() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(TransportError):
        patient(httpx.MockTransport(handler), max_attempts=2).answer("q?", [doc("a")])
    assert calls == 2


def test_a_rejected_key_is_not_retried() -> None:
    """A rejected key stays rejected; retrying only delays a clear message."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(SynthesisUnavailable):
        patient(httpx.MockTransport(handler)).answer("q?", [doc("a")])
    assert calls == 1


def test_a_retired_model_is_not_retried() -> None:
    """Counting completions only: a 404 also fetches the model list to report it."""
    completions = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completions
        if request.url.path.endswith("/chat/completions"):
            completions += 1
        return httpx.Response(404, json={"error": "no model"})

    with pytest.raises(SynthesisUnavailable):
        patient(httpx.MockTransport(handler)).answer("q?", [doc("a")])
    assert completions == 1


def test_a_client_error_that_is_not_transient_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="malformed")

    with pytest.raises(TransportError):
        patient(httpx.MockTransport(handler)).answer("q?", [doc("a")])
    assert calls == 1


def test_retry_after_is_honoured() -> None:
    waits: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        if not waits:
            return httpx.Response(429, text="slow", headers={"retry-after": "2"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok [1]"}}]})

    synth = Synthesiser(
        api_key="k",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=waits.append,
    )
    synth.answer("q?", [doc("a")])
    assert waits == [2.0]
