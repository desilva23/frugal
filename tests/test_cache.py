"""Tests for the record/replay cache.

The cache decides what Frugal pays for, so these lean hard on the cases that
would quietly cost money or quietly invalidate a benchmark: key instability,
credential leakage, stale fixtures, and partial writes.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from frugal.cache import (
    CacheEntry,
    CacheMode,
    ResponseCache,
    canonicalise_params,
    compute_key,
    redact_params,
)
from frugal.errors import CacheCorrupt, CacheMiss

RESPONSE = {"organic_results": [{"title": "result", "link": "https://example.com"}]}


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache")


# --------------------------------------------------------------------------
# Key derivation
# --------------------------------------------------------------------------


def test_key_is_stable_across_parameter_order() -> None:
    """Reordering parameters must not produce a second billable search."""
    assert compute_key("google", {"q": "x", "hl": "en"}) == compute_key(
        "google", {"hl": "en", "q": "x"}
    )


def test_key_ignores_none_values() -> None:
    """Passing num=None must match omitting num; SerpApi drops it either way."""
    assert compute_key("google", {"q": "x", "num": None}) == compute_key("google", {"q": "x"})


def test_key_ignores_credentials() -> None:
    """The same search with two different keys is one search, not two."""
    assert compute_key("google", {"q": "x", "api_key": "aaa"}) == compute_key(
        "google", {"q": "x", "api_key": "bbb"}
    )


def test_key_ignores_credential_name_casing() -> None:
    assert compute_key("google", {"q": "x", "API_Key": "aaa"}) == compute_key("google", {"q": "x"})


def test_key_separates_engines() -> None:
    """Identical parameters on different engines are different searches."""
    assert compute_key("google", {"q": "x"}) != compute_key("bing", {"q": "x"})


def test_key_normalises_engine_casing() -> None:
    assert compute_key("Google", {"q": "x"}) == compute_key("google", {"q": "x"})


def test_key_distinguishes_different_queries() -> None:
    assert compute_key("google", {"q": "x"}) != compute_key("google", {"q": "y"})


def test_booleans_coerce_before_int_branch() -> None:
    """bool subclasses int; it must serialise as SerpApi's string form."""
    assert canonicalise_params({"safe": True})["safe"] == "true"
    assert canonicalise_params({"safe": False})["safe"] == "false"
    assert canonicalise_params({"num": 1})["num"] == 1


def test_nested_structures_canonicalise() -> None:
    params = canonicalise_params({"filters": {"b": 2, "a": [1, True]}})
    assert params["filters"] == {"a": [1, "true"], "b": 2}


def test_unsupported_value_type_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(TypeError, match="cannot be part of a cache key"):
        compute_key("google", {"when": datetime.now(UTC)})


def test_redact_keeps_key_present_but_hides_value() -> None:
    redacted = redact_params({"q": "x", "api_key": "secret"})
    assert redacted["q"] == "x"
    assert redacted["api_key"] == "<redacted>"


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_store_then_load_round_trips(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    entry = cache.load("google", {"q": "x"})
    assert entry is not None
    assert entry.response == RESPONSE
    assert entry.engine == "google"


def test_stored_record_never_contains_the_credential(cache: ResponseCache) -> None:
    """The single most important property here: keys must not reach disk."""
    cache.store("google", {"q": "x", "api_key": "super-secret-value"}, RESPONSE)
    on_disk = (cache.directory).rglob("*.json")
    contents = "\n".join(p.read_text(encoding="utf-8") for p in on_disk)
    assert "super-secret-value" not in contents
    assert "<redacted>" in contents


def test_entry_is_written_as_readable_json(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    path = next(cache.directory.rglob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["frugal_cache_version"] == 1
    assert record["response"] == RESPONSE


def test_entries_are_sharded_by_digest_prefix(cache: ResponseCache) -> None:
    key = compute_key("google", {"q": "x"})
    path = cache.path_for("google", key)
    assert path.parent.name == key[:2]
    assert path.parent.parent.name == "google"


def test_engine_names_are_sanitised_into_path_segments(cache: ResponseCache) -> None:
    path = cache.path_for("weird/../engine", compute_key("x", {}))
    assert "/.." not in str(path)
    assert path.is_relative_to(cache.directory)


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------


def test_fresh_entry_hits(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    assert cache.load("google", {"q": "x"}) is not None
    assert cache.stats.hits == 1


def test_stale_entry_misses_and_is_counted(cache: ResponseCache) -> None:
    old = datetime.now(UTC) - timedelta(days=30)
    cache.store("google", {"q": "x"}, RESPONSE, fetched_at=old)
    assert cache.load("google", {"q": "x"}) is None
    assert cache.stats.stale == 1
    assert cache.stats.hits == 0


def test_ttl_varies_by_engine_volatility(cache: ResponseCache) -> None:
    assert cache.ttl_for("google_news") < cache.ttl_for("google")
    assert cache.ttl_for("google") < cache.ttl_for("google_patents")


def test_unknown_engine_uses_fallback_ttl(cache: ResponseCache) -> None:
    assert cache.ttl_for("some_new_engine") == cache.fallback_ttl


def test_future_timestamp_does_not_read_as_infinitely_fresh() -> None:
    """A fixture captured on a clock that ran ahead must not be immortal."""
    entry = CacheEntry(
        engine="google",
        key="k",
        params={},
        response={},
        fetched_at=datetime.now(UTC) + timedelta(days=1),
        elapsed_ms=0.0,
        searches_charged=1,
    )
    assert entry.age_seconds() == 0.0


def test_naive_timestamps_are_treated_as_utc(tmp_path: Path) -> None:
    """Hand-edited fixtures often lose the timezone suffix."""
    record = {
        "frugal_cache_version": 1,
        "engine": "google",
        "key": "k",
        "params": {},
        "response": {},
        "fetched_at": "2026-09-16T12:00:00",
        "elapsed_ms": 1.0,
        "searches_charged": 1,
    }
    entry = CacheEntry.from_record(record, path=tmp_path / "x.json")
    assert entry.fetched_at.tzinfo is UTC


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def test_replay_raises_on_miss_rather_than_reaching_for_the_network(tmp_path: Path) -> None:
    replay = ResponseCache(tmp_path / "cache", mode=CacheMode.REPLAY)
    with pytest.raises(CacheMiss) as excinfo:
        replay.load("google", {"q": "absent"})
    assert "re-record" in str(excinfo.value)


def test_replay_ignores_ttl_so_recorded_benchmarks_stay_reproducible(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    ResponseCache(directory).store(
        "google_news", {"q": "x"}, RESPONSE, fetched_at=datetime.now(UTC) - timedelta(days=365)
    )
    replay = ResponseCache(directory, mode=CacheMode.REPLAY)
    assert replay.load("google_news", {"q": "x"}) is not None


def test_record_mode_always_misses_so_the_caller_refetches(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    ResponseCache(directory).store("google", {"q": "x"}, RESPONSE)
    recorder = ResponseCache(directory, mode=CacheMode.RECORD)
    assert recorder.load("google", {"q": "x"}) is None


# --------------------------------------------------------------------------
# Damaged entries
# --------------------------------------------------------------------------


def _corrupt_the_only_entry(cache: ResponseCache, text: str = "{not json") -> None:
    next(cache.directory.rglob("*.json")).write_text(text, encoding="utf-8")


def test_corrupt_entry_is_a_miss_under_auto(cache: ResponseCache) -> None:
    """One bad file should cost a single search to repair, not abort the run."""
    cache.store("google", {"q": "x"}, RESPONSE)
    _corrupt_the_only_entry(cache)
    assert cache.load("google", {"q": "x"}) is None
    assert cache.stats.corrupt == 1


def test_corrupt_entry_raises_under_replay(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    writer = ResponseCache(directory)
    writer.store("google", {"q": "x"}, RESPONSE)
    _corrupt_the_only_entry(writer)
    with pytest.raises(CacheCorrupt):
        ResponseCache(directory, mode=CacheMode.REPLAY).load("google", {"q": "x"})


def test_unknown_format_version_is_corrupt_not_a_crash(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    _corrupt_the_only_entry(cache, json.dumps({"frugal_cache_version": 999}))
    assert cache.load("google", {"q": "x"}) is None


def test_json_scalar_at_top_level_is_corrupt_not_a_crash(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    _corrupt_the_only_entry(cache, "42")
    assert cache.load("google", {"q": "x"}) is None


def test_iter_entries_skips_damaged_files_without_stopping(cache: ResponseCache) -> None:
    cache.store("google", {"q": "good"}, RESPONSE)
    cache.store("google", {"q": "bad"}, RESPONSE)
    bad = cache.path_for("google", compute_key("google", {"q": "bad"}))
    bad.write_text("{{{", encoding="utf-8")
    assert [e.params["q"] for e in cache.iter_entries()] == ["good"]


# --------------------------------------------------------------------------
# Durability and concurrency
# --------------------------------------------------------------------------


def test_write_leaves_no_temporary_files_behind(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    assert list(cache.directory.rglob("*.tmp")) == []


def test_overwrite_replaces_cleanly(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, {"v": 1})
    cache.store("google", {"q": "x"}, {"v": 2})
    entry = cache.load("google", {"q": "x"})
    assert entry is not None and entry.response == {"v": 2}
    assert len(list(cache.directory.rglob("*.json"))) == 1


def test_concurrent_writes_to_one_key_produce_one_valid_entry(cache: ResponseCache) -> None:
    """os.replace is atomic, so readers see one version or the other, never both."""
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            cache.store("google", {"q": "x"}, {"writer": n})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(list(cache.directory.rglob("*.json"))) == 1
    entry = cache.load("google", {"q": "x"})
    assert entry is not None and "writer" in entry.response


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


def test_stats_track_spend_and_hit_rate(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    cache.load("google", {"q": "x"})
    cache.load("google", {"q": "never-stored"})
    assert cache.stats.searches_charged == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


def test_hit_rate_is_zero_before_any_lookup(cache: ResponseCache) -> None:
    assert cache.stats.hit_rate == 0.0
    assert cache.stats.as_dict()["lookups"] == 0


def test_multi_search_responses_charge_accordingly(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE, searches_charged=5)
    assert cache.stats.searches_charged == 5


# --------------------------------------------------------------------------
# peek: is this recorded?
# --------------------------------------------------------------------------


def test_peek_finds_a_recorded_entry(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    assert cache.peek("google", {"q": "x"}) is not None


def test_peek_returns_none_for_an_absent_entry(cache: ResponseCache) -> None:
    assert cache.peek("google", {"q": "never stored"}) is None


def test_peek_does_not_raise_under_replay(tmp_path: Path) -> None:
    """A peek asks "is it there?", which has an answer in every mode.

    Routing it through load() made it raise CacheMiss under REPLAY, which took
    down a caller that was only trying to find out whether a search was free.
    """
    replay = ResponseCache(tmp_path / "cache", mode=CacheMode.REPLAY)
    assert replay.peek("google", {"q": "absent"}) is None


def test_peek_ignores_ttl(cache: ResponseCache) -> None:
    """An expired entry is still recorded, and still free to serve."""
    old = datetime.now(UTC) - timedelta(days=365)
    cache.store("google_news", {"q": "x"}, RESPONSE, fetched_at=old)
    assert cache.peek("google_news", {"q": "x"}) is not None


def test_peek_does_not_move_the_statistics(cache: ResponseCache) -> None:
    """The hit rate is a reported number; peeking must not inflate it."""
    cache.store("google", {"q": "x"}, RESPONSE)
    before = cache.stats.as_dict()
    cache.peek("google", {"q": "x"})
    cache.peek("google", {"q": "absent"})
    assert cache.stats.as_dict() == before


def test_peek_reports_a_corrupt_entry_as_absent(cache: ResponseCache) -> None:
    cache.store("google", {"q": "x"}, RESPONSE)
    next(cache.directory.rglob("*.json")).write_text("{{{", encoding="utf-8")
    assert cache.peek("google", {"q": "x"}) is None
