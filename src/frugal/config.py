"""Credential and configuration resolution.

Frugal reads its key from a ``.env`` file in the project root rather than
requiring an ``export`` in every new shell. The parser here is deliberately
small and dependency-free, but it tolerates the ways people actually write these
files: ``export`` prefixes pasted from documentation, quoted values, inline
comments, CRLF endings from an editor on another platform, and a byte-order mark
from one that helpfully added it.

Resolution order is explicit argument, then process environment, then ``.env``.
The process environment wins over the file so that CI and one-off overrides
behave the way anyone would expect, matching the convention every other dotenv
implementation follows.
"""

from __future__ import annotations

import os
from pathlib import Path

from frugal.errors import MissingAPIKey

ENV_VAR = "SERPAPI_API_KEY"

#: Answer synthesis is optional, so this key is too. Nothing in the benchmark
#: reads it.
SYNTHESIS_ENV_VAR = "GROQ_API_KEY"

DOTENV_NAME = ".env"

#: How far up the tree to look for a .env before giving up. Deep enough to find
#: the project root from a nested test or benchmark directory, shallow enough
#: never to wander into a home directory and read someone else's file.
_MAX_PARENTS = 6


def find_dotenv(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate the nearest ``.env``, walking upward from ``start``.

    Returns ``None`` rather than raising when there is none: a missing file is
    an ordinary state, since the key may legitimately come from the environment.
    """
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *list(current.parents)[:_MAX_PARENTS]]:
        dotenv = candidate / DOTENV_NAME
        if dotenv.is_file():
            return dotenv
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse dotenv-format ``text`` into a mapping.

    Unparseable lines are skipped rather than raising. A stray line in a config
    file should not stop a program that does not need the value on it.
    """
    values: dict[str, str] = {}

    # An editor on Windows may leave a BOM, which would otherwise become part of
    # the first key's name and make it silently unmatchable.
    text = text.lstrip("﻿")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # People paste `export FOO=bar` straight out of documentation.
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        name, separator, raw_value = line.partition("=")
        if not separator:
            continue

        name = name.strip()
        if not name:
            continue

        values[name] = _parse_value(raw_value.strip())

    return values


def _parse_value(value: str) -> str:
    """Strip quoting and any trailing comment from a dotenv value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        # Quoted: everything inside is literal, including a '#' that would
        # otherwise read as a comment.
        return value[1:-1]

    # Unquoted: a '#' preceded by whitespace starts a comment. Requiring the
    # whitespace keeps keys that legitimately contain '#' intact.
    for index, char in enumerate(value):
        if char == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].strip()
    return value.strip()


def load_dotenv(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Read and parse a ``.env``, returning ``{}`` when there is none to read.

    Does not mutate :data:`os.environ`. Callers that want the values in the
    process environment can set them; keeping this function pure makes it
    testable without leaking state between tests.
    """
    dotenv = Path(path) if path is not None else find_dotenv()
    if dotenv is None or not Path(dotenv).is_file():
        return {}
    try:
        return parse_dotenv(Path(dotenv).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        # An unreadable or binary .env is equivalent to none for our purposes;
        # the caller will raise MissingAPIKey with actionable instructions.
        return {}


def resolve_key(
    variable: str,
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> str:
    """Return the value of ``variable``, or raise explaining how to set it.

    A key that is present but blank counts as absent. Someone who has created a
    ``.env`` and left the placeholder in place has not configured anything, and
    telling them the key is missing is more useful than letting the API reject
    it later with a 401.
    """
    environment = os.environ if env is None else env

    if explicit and explicit.strip():
        return explicit.strip()

    from_env = environment.get(variable, "")
    if from_env.strip() and not _is_placeholder(from_env):
        return from_env.strip()

    dotenv = Path(dotenv_path) if dotenv_path is not None else find_dotenv()
    from_file = load_dotenv(dotenv).get(variable, "")
    if from_file.strip() and not _is_placeholder(from_file):
        return from_file.strip()

    searched = [f"${variable}", str(dotenv) if dotenv else f"{DOTENV_NAME} (not found)"]
    raise MissingAPIKey(searched)


def resolve_api_key(
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> str:
    """Return the SerpApi key, or raise :class:`MissingAPIKey`."""
    return resolve_key(ENV_VAR, explicit, env=env, dotenv_path=dotenv_path)


def _is_placeholder(value: str) -> bool:
    """Whether ``value`` is the shipped template text rather than a real key."""
    return value.strip().lower() in {"your-key-here", "your_key_here", "changeme", "xxx"}


def describe_key_source(
    *,
    env: dict[str, str] | None = None,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> tuple[bool, str]:
    """Report whether a key is configured and where it came from.

    Powers ``frugal doctor``. Returns a flag and a human-readable line; never
    returns the key itself, so the output is safe to paste into an issue.
    """
    environment = os.environ if env is None else env
    dotenv = Path(dotenv_path) if dotenv_path is not None else find_dotenv()

    from_env = environment.get(ENV_VAR, "")
    if from_env.strip() and not _is_placeholder(from_env):
        return True, f"found in ${ENV_VAR} ({_mask(from_env.strip())})"

    from_file = load_dotenv(dotenv).get(ENV_VAR, "")
    if from_file.strip() and not _is_placeholder(from_file):
        return True, f"found in {dotenv} ({_mask(from_file.strip())})"

    if from_env.strip() or from_file.strip():
        location = f"${ENV_VAR}" if from_env.strip() else str(dotenv)
        return False, f"{location} still contains the placeholder value"

    if dotenv is None:
        return False, f"no {DOTENV_NAME} found and ${ENV_VAR} is unset"
    return False, f"{dotenv} exists but does not set {ENV_VAR}"


def _mask(key: str) -> str:
    """Render a key for display without disclosing it."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 8}{key[-4:]}"
