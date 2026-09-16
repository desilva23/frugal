"""Turning retrieved evidence into a written answer.

Deliberately outside everything the benchmark measures. Routing, reformulation,
budgeting and the stopping rule are all deterministic so that a published result
table reproduces without a key or a model; putting a language model anywhere in
that path would destroy the property. Synthesis runs after the plan has
finished, reads only what the plan already retrieved, and changes no number in
the results table. It can be switched off entirely and the project's claims are
unaffected.

What it adds is the difference between a list of links and an answer. The
planner's job is to have the right evidence in hand; this reads it out.

Three rules are enforced rather than requested. The model sees only the
retrieved evidence, so an answer it invents from memory has nothing to cite.
Citations are validated against the evidence actually supplied, and invented
ones are stripped rather than shown. And when the evidence does not answer the
question, saying so is the correct output — a confident answer assembled from
irrelevant sources is worse than an admission.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from frugal.config import SYNTHESIS_ENV_VAR, resolve_key
from frugal.errors import FrugalError, TransportError
from frugal.schema import Document, Evidence, Series

#: Groq speaks the OpenAI chat-completions dialect, so this endpoint shape works
#: for any provider that does. Point ``base_url`` elsewhere to use another.
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

#: Overridable, because hosted model names change more often than this code
#: will. The previous default was retired between one week and the next, which
#: is why a 404 now lists what the account can actually reach rather than
#: leaving the reader to go and look.
DEFAULT_MODEL = "openai/gpt-oss-120b"

DEFAULT_TIMEOUT = 60.0

#: Evidence items shown to the model. Beyond this the prompt grows without
#: improving the answer, and the planner's own depth limit means later items are
#: the weakest anyway.
MAX_EVIDENCE = 20

#: Snippets are truncated rather than dropped: the title and source carry most
#: of the signal, and a long snippet crowds out a whole other result.
MAX_SNIPPET = 320

_CITATION = re.compile(r"\[(\d+)\]")

#: Bracket characters models use in place of ASCII ones. Observed in the wild:
#: gpt-oss-120b returned CJK full-width brackets, which the citation pattern did
#: not match, so a correctly cited answer was reported as ungrounded -- a false
#: accusation of hallucination, which is worse than missing a citation.
#: Built from code points and named rather than written literally, since a block
#: of look-alike brackets is unreadable and ruff is right to object to it.
_BRACKET_ALIASES: dict[str, str | int | None] = {
    chr(0x3010): "[",  # CJK left black lenticular
    chr(0x3011): "]",  # CJK right black lenticular
    chr(0xFF3B): "[",  # fullwidth left square
    chr(0xFF3D): "]",  # fullwidth right square
    chr(0x3014): "[",  # left tortoise shell
    chr(0x3015): "]",  # right tortoise shell
    chr(0x2768): "[",  # medium left parenthesis ornament
    chr(0x2769): "]",  # medium right parenthesis ornament
}


def normalise_brackets(text: str) -> str:
    """Rewrite non-ASCII bracket characters so citations can be recognised."""
    return text.translate(str.maketrans(_BRACKET_ALIASES))


_SYSTEM_PROMPT = """You answer questions using only the numbered evidence provided.

Rules:
- Use only the evidence given. Do not add facts from your own knowledge.
- Cite every claim with the evidence number in plain ASCII square brackets, \
like [3]. Do not use any other bracket characters.
- If the evidence does not answer the question, say so plainly and explain what \
is missing. Do not guess.
- Be concise: two to four sentences unless the question needs more.
- When a time series is provided, state the direction and magnitude of change \
and cite it."""


class SynthesisUnavailable(FrugalError):
    """Synthesis was asked for but cannot run.

    Separate from a transport failure because the remedy is different: this
    means the feature was never configured, and the rest of the project works
    without it.
    """


@dataclass(frozen=True, slots=True)
class Citation:
    """One evidence item an answer referred to."""

    number: int
    title: str
    url: str | None
    engine: str

    def describe(self) -> str:
        return f"[{self.number}] {self.title}" + (f" — {self.url}" if self.url else "")


@dataclass(frozen=True, slots=True)
class Answer:
    """A written answer and the evidence it rests on."""

    text: str
    citations: tuple[Citation, ...]
    model: str
    evidence_offered: int
    elapsed_ms: float

    @property
    def grounded(self) -> bool:
        """Whether the answer cited anything at all.

        An uncited answer is either a refusal, which is fine, or an answer from
        the model's own memory, which is not. Either way the caller should know.
        """
        return bool(self.citations)


def render_evidence(evidence: Sequence[Evidence], *, limit: int = MAX_EVIDENCE) -> str:
    """Format evidence as a numbered list for the prompt.

    A series is rendered as its summary line rather than its raw points: the
    direction and magnitude are what answer a question, and fifty timestamped
    numbers would crowd out every other result.
    """
    lines: list[str] = []
    for number, item in enumerate(evidence[:limit], start=1):
        if isinstance(item, Series):
            lines.append(f"[{number}] TIME SERIES — {item.summarise()} (source: google_trends)")
            continue
        if not isinstance(item, Document):  # pragma: no cover - Evidence is a closed union
            continue
        parts = [f"[{number}] {item.title}"]
        if item.source:
            parts.append(f"source: {item.source}")
        if item.url:
            parts.append(item.url)
        if item.snippet:
            snippet = item.snippet[:MAX_SNIPPET]
            if len(item.snippet) > MAX_SNIPPET:
                snippet += "…"
            parts.append(snippet)
        lines.append("\n    ".join(parts))
    return "\n".join(lines)


def extract_citations(text: str, evidence: Sequence[Evidence]) -> tuple[str, tuple[Citation, ...]]:
    """Validate the citations in ``text`` against ``evidence``.

    A number the model invented refers to nothing, so it is removed from the
    text rather than rendered as a citation the reader cannot follow. Returns
    the cleaned text and the citations that were real.
    """
    text = normalise_brackets(text)
    found: dict[int, Citation] = {}
    invalid: set[str] = set()

    for match in _CITATION.finditer(text):
        raw = match.group(1)
        number = int(raw)
        if not 1 <= number <= len(evidence):
            invalid.add(match.group(0))
            continue
        if number in found:
            continue
        item = evidence[number - 1]
        found[number] = Citation(
            number=number,
            title=item.name if isinstance(item, Series) else item.title,
            url=None if isinstance(item, Series) else item.url,
            engine=item.provenance.engine,
        )

    cleaned = text
    for token in invalid:
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    return cleaned, tuple(found[key] for key in sorted(found))


class Synthesiser:
    """Writes an answer from retrieved evidence.

    Owns its HTTP client unless one is injected, which is what lets the tests
    drive it with a mock transport and no network.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client(timeout=timeout)

        try:
            self._api_key = resolve_key(SYNTHESIS_ENV_VAR, api_key)
        except FrugalError as exc:
            raise SynthesisUnavailable(
                f"answer synthesis needs a {SYNTHESIS_ENV_VAR}.\n\n"
                f"Add this line to your .env:\n\n"
                f"    {SYNTHESIS_ENV_VAR}=your-key-here\n\n"
                f"A free key is available at https://console.groq.com/keys\n"
                f"Everything else in Frugal works without it."
            ) from exc

    def answer(self, question: str, evidence: Sequence[Evidence]) -> Answer:
        """Write an answer to ``question`` from ``evidence``."""
        started = time.perf_counter()
        offered = list(evidence[:MAX_EVIDENCE])

        if not offered:
            # Calling a model with nothing to read can only produce an answer
            # from its own memory, which is the failure this exists to avoid.
            return Answer(
                text="No evidence was retrieved, so there is nothing to answer from.",
                citations=(),
                model=self.model,
                evidence_offered=0,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{render_evidence(offered)}\n\n"
            f"Answer the question using only this evidence, citing by number."
        )
        raw = self._complete(prompt)
        text, citations = extract_citations(raw, offered)

        return Answer(
            text=text,
            citations=citations,
            model=self.model,
            evidence_offered=len(offered),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def _complete(self, prompt: str) -> str:
        """One chat completion, with the failure modes named."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            # Low rather than zero: some providers reject zero outright, and a
            # citation-bound answer has little room to wander regardless.
            "temperature": 0.1,
        }

        try:
            response = self._http.post(
                self.base_url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise TransportError(f"could not reach the synthesis endpoint: {exc}") from exc

        if response.status_code == 401:
            raise SynthesisUnavailable(
                f"the synthesis endpoint rejected the key. Check {SYNTHESIS_ENV_VAR} in your .env."
            )
        if response.status_code == 404:
            available = self._available_models()
            listing = ""
            if available:
                names = "\n".join(f"  {name}" for name in available)
                listing = f"\n\nAvailable on this account:\n{names}"
            raise SynthesisUnavailable(
                f"the model {self.model!r} was not found. Hosted model names change "
                f"often; pass --model with a current one.{listing}"
            )
        if response.is_error:
            raise TransportError(
                f"synthesis failed with HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TransportError(
                f"unexpected response from the synthesis endpoint: {response.text[:200]}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise TransportError("the synthesis endpoint returned an empty answer")
        return content.strip()

    def _available_models(self) -> list[str]:
        """Chat models this account can reach, for the not-found message.

        Best effort: this runs while already reporting a failure, so it must not
        raise one of its own. Audio and safety models are filtered out because
        they cannot answer a question and listing them would send the reader
        down a blind alley.
        """
        try:
            response = self._http.get(
                self.base_url.rsplit("/", 1)[0] + "/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.is_error:
                return []
            names = [str(entry["id"]) for entry in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

        skip = ("whisper", "guard", "orpheus", "tts", "safeguard")
        return sorted(n for n in names if not any(word in n.lower() for word in skip))

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Synthesiser:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
