"""Tests for credential resolution.

Weighted toward the ways a .env actually arrives in the wild: pasted with an
`export` prefix, quoted, commented, saved by an editor that added a BOM or CRLF
line endings, or created from the template and never filled in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frugal.config import (
    ENV_VAR,
    describe_key_source,
    find_dotenv,
    load_dotenv,
    parse_dotenv,
    resolve_api_key,
)
from frugal.errors import MissingAPIKey

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_a_plain_assignment() -> None:
    assert parse_dotenv("SERPAPI_API_KEY=abc123") == {"SERPAPI_API_KEY": "abc123"}


def test_tolerates_an_export_prefix_pasted_from_documentation() -> None:
    assert parse_dotenv("export SERPAPI_API_KEY=abc123")["SERPAPI_API_KEY"] == "abc123"


def test_strips_surrounding_quotes() -> None:
    assert parse_dotenv('SERPAPI_API_KEY="abc123"')["SERPAPI_API_KEY"] == "abc123"
    assert parse_dotenv("SERPAPI_API_KEY='abc123'")["SERPAPI_API_KEY"] == "abc123"


def test_ignores_comments_and_blank_lines() -> None:
    text = "# a comment\n\nSERPAPI_API_KEY=abc123\n\n# another\n"
    assert parse_dotenv(text) == {"SERPAPI_API_KEY": "abc123"}


def test_strips_an_inline_comment() -> None:
    assert parse_dotenv("SERPAPI_API_KEY=abc123  # my key")["SERPAPI_API_KEY"] == "abc123"


def test_keeps_a_hash_that_is_part_of_the_value() -> None:
    """A '#' only starts a comment when whitespace precedes it."""
    assert parse_dotenv("SERPAPI_API_KEY=abc#123")["SERPAPI_API_KEY"] == "abc#123"


def test_keeps_a_hash_inside_quotes() -> None:
    assert parse_dotenv('SERPAPI_API_KEY="abc #123"')["SERPAPI_API_KEY"] == "abc #123"


def test_keeps_equals_signs_inside_the_value() -> None:
    assert parse_dotenv("SERPAPI_API_KEY=abc=123==")["SERPAPI_API_KEY"] == "abc=123=="


def test_tolerates_whitespace_around_the_assignment() -> None:
    assert parse_dotenv("  SERPAPI_API_KEY  =  abc123  ")["SERPAPI_API_KEY"] == "abc123"


def test_tolerates_crlf_line_endings() -> None:
    assert parse_dotenv("A=1\r\nSERPAPI_API_KEY=abc123\r\n")["SERPAPI_API_KEY"] == "abc123"


def test_tolerates_a_byte_order_mark() -> None:
    """A BOM would otherwise become part of the first key's name."""
    assert parse_dotenv("﻿SERPAPI_API_KEY=abc123") == {"SERPAPI_API_KEY": "abc123"}


def test_skips_unparseable_lines_without_raising() -> None:
    assert parse_dotenv("this is not an assignment\nSERPAPI_API_KEY=abc123") == {
        "SERPAPI_API_KEY": "abc123"
    }


def test_skips_an_assignment_with_no_name() -> None:
    assert parse_dotenv("=value") == {}


def test_empty_value_parses_as_empty_string() -> None:
    assert parse_dotenv("SERPAPI_API_KEY=") == {"SERPAPI_API_KEY": ""}


# --------------------------------------------------------------------------
# Locating the file
# --------------------------------------------------------------------------


def test_finds_a_dotenv_in_a_parent_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SERPAPI_API_KEY=abc123", encoding="utf-8")
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"


def test_returns_none_when_there_is_no_dotenv(tmp_path: Path) -> None:
    assert find_dotenv(tmp_path) is None


def test_load_returns_empty_mapping_when_absent(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nonexistent.env") == {}


def test_load_returns_empty_mapping_for_an_unreadable_file(tmp_path: Path) -> None:
    binary = tmp_path / ".env"
    binary.write_bytes(b"\xff\xfe\x00\x00binary garbage")
    assert load_dotenv(binary) == {}


# --------------------------------------------------------------------------
# Resolution order
# --------------------------------------------------------------------------


def test_explicit_argument_wins(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{ENV_VAR}=from-file", encoding="utf-8")
    resolved = resolve_api_key("explicit", env={ENV_VAR: "from-env"}, dotenv_path=dotenv)
    assert resolved == "explicit"


def test_environment_beats_the_file(tmp_path: Path) -> None:
    """Matches every other dotenv implementation, and is what CI expects."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{ENV_VAR}=from-file", encoding="utf-8")
    assert resolve_api_key(env={ENV_VAR: "from-env"}, dotenv_path=dotenv) == "from-env"


def test_file_is_used_when_the_environment_is_unset(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{ENV_VAR}=from-file", encoding="utf-8")
    assert resolve_api_key(env={}, dotenv_path=dotenv) == "from-file"


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    assert resolve_api_key(env={ENV_VAR: "  abc123  "}, dotenv_path=tmp_path / "none") == "abc123"


def test_blank_value_counts_as_missing(tmp_path: Path) -> None:
    with pytest.raises(MissingAPIKey):
        resolve_api_key(env={ENV_VAR: "   "}, dotenv_path=tmp_path / "none")


def test_unfilled_template_counts_as_missing(tmp_path: Path) -> None:
    """Copying .env.example and not editing it is not configuration."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{ENV_VAR}=your-key-here", encoding="utf-8")
    with pytest.raises(MissingAPIKey):
        resolve_api_key(env={}, dotenv_path=dotenv)


def test_missing_key_error_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(MissingAPIKey) as excinfo:
        resolve_api_key(env={}, dotenv_path=tmp_path / "none")
    message = str(excinfo.value)
    assert ".env" in message
    assert "serpapi.com" in message
    assert ENV_VAR in message


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_describe_reports_the_source_without_disclosing_the_key(tmp_path: Path) -> None:
    ok, detail = describe_key_source(
        env={ENV_VAR: "abcd1234567890wxyz"}, dotenv_path=tmp_path / "none"
    )
    assert ok
    assert "abcd1234567890wxyz" not in detail
    assert "abcd" in detail and "wxyz" in detail


def test_describe_masks_a_short_key_entirely(tmp_path: Path) -> None:
    _, detail = describe_key_source(env={ENV_VAR: "abc"}, dotenv_path=tmp_path / "none")
    assert "abc" not in detail


def test_describe_calls_out_an_unfilled_placeholder(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{ENV_VAR}=your-key-here", encoding="utf-8")
    ok, detail = describe_key_source(env={}, dotenv_path=dotenv)
    assert not ok
    assert "placeholder" in detail


def test_describe_reports_a_missing_file(tmp_path: Path) -> None:
    ok, detail = describe_key_source(env={}, dotenv_path=tmp_path / "none")
    assert not ok
    assert "unset" in detail or "does not set" in detail
