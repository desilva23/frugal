"""The repository's scripts, where a mistake costs data rather than a number."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _export() -> dict[str, object]:
    return runpy.run_path(str(ROOT / "scripts" / "export_fixtures.py"), run_name="export")


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        ("fixtures", "cache"),            # swapped arguments
        ("cache", "."),                   # the repository root
        ("cache", "benchmarks"),          # the question sets and results
        ("cache", "cache/fixtures"),      # inside the source
    ],
)
def test_the_export_refuses_to_delete_anything_but_a_fixture_set(
    tmp_path: Path, source: str, destination: str
) -> None:
    """The destination is deleted and rewritten, so it must be a fixture set.

    It became a positional argument when the repository started carrying two
    recordings; swapped arguments would have deleted the live cache.
    """
    (tmp_path / "cache" / "google").mkdir(parents=True)
    (tmp_path / "cache" / "google" / "a.json").write_text('{"response": {}}')
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "questions.json").write_text("{}")

    before = sorted(str(p) for p in tmp_path.rglob("*"))
    code = _export()["export"](tmp_path / source, tmp_path / destination)  # type: ignore[operator]
    assert code == 1
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_the_export_writes_a_fixture_set(tmp_path: Path) -> None:
    (tmp_path / "cache" / "google").mkdir(parents=True)
    (tmp_path / "cache" / "google" / "a.json").write_text('{"response": {"organic_results": []}}')
    code = _export()["export"](tmp_path / "cache", tmp_path / "fixtures-new")  # type: ignore[operator]
    assert code == 0
    assert (tmp_path / "fixtures-new" / "google" / "a.json").exists()
