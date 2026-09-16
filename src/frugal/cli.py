"""Command-line entry point.

``frugal doctor`` exists because the first thing that goes wrong for anyone
trying a project is credentials, and "401 Unauthorized" three layers down a
stack trace is a poor way to learn that a ``.env`` was never created.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from frugal import __version__
from frugal.cache import CacheMode, ResponseCache
from frugal.config import DOTENV_NAME, ENV_VAR, describe_key_source, find_dotenv

DEFAULT_CACHE_DIR = ".frugal-cache"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frugal",
        description="A cost-aware search planner for SerpApi.",
    )
    parser.add_argument("--version", action="version", version=f"frugal {__version__}")

    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="check that the local setup is ready to run")

    return parser


def doctor() -> int:
    """Report on local setup. Returns a process exit code."""
    ok, detail = describe_key_source()

    print(f"frugal {__version__}")
    print(f"python  {sys.version.split()[0]}")
    print()
    print(f"[{'ok ' if ok else 'xx '}] SerpApi key: {detail}")

    cache = ResponseCache(DEFAULT_CACHE_DIR, mode=CacheMode.AUTO)
    entries = sum(1 for _ in cache.iter_entries()) if cache.directory.exists() else 0
    noun = "entry" if entries == 1 else "entries"
    print(f"[ok ] cache: {cache.directory} ({entries} recorded {noun})")

    if not ok:
        dotenv = find_dotenv()
        target = dotenv or f"{DOTENV_NAME} (create it in the project root)"
        print()
        print("To fix:")
        print("  1. Get a free key at https://serpapi.com/manage-api-key")
        print(f"  2. Put this line in {target}:")
        print(f"         {ENV_VAR}=your-actual-key")
        print()
        print(f"  {DOTENV_NAME} is gitignored, so the key will not be committed.")
        return 1

    print()
    print("Setup looks good.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a command. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    return 1  # pragma: no cover - argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
