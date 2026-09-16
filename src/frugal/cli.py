"""Command-line interface.

``frugal ask`` is the face of the project. Everything it prints is the planner's
own trace rather than decoration: the routing scores are what selected the
engines, the cost counter is the budget governor, and the saturation line is the
stopping rule explaining itself. If a number here is wrong, the planner is
wrong.

``frugal plan`` shows the same plan without issuing anything, which is the
clearest demonstration of what a cost-aware planner is for — you can see what a
question will cost before you agree to pay it.

``frugal doctor`` exists because credentials are the first thing that goes wrong
for anyone trying a project, and "401 Unauthorized" three layers down a stack
trace is a poor way to learn that a ``.env`` was never created.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from frugal import __version__
from frugal.cache import CacheMode, ResponseCache
from frugal.client import SerpApiClient
from frugal.config import DOTENV_NAME, ENV_VAR, describe_key_source, find_dotenv
from frugal.errors import FrugalError
from frugal.planner import DEFAULT_BUDGET, DEFAULT_MAX_ENGINES, Planner, PlanResult
from frugal.router import route
from frugal.schema import Document, Series

DEFAULT_CACHE_DIR = ".frugal-cache"

#: Unicode blocks for inline sparklines. A series is the one kind of evidence a
#: web search cannot return, so it is worth showing as a shape rather than a
#: sentence.
_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    """Render numbers as a single line of block characters."""
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return _BLOCKS[len(_BLOCKS) // 2] * len(values)
    span = high - low
    return "".join(_BLOCKS[int((v - low) / span * (len(_BLOCKS) - 1))] for v in values)


def _routing_table(question: str, limit: int) -> Table:
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(justify="right", style="yellow")
    table.add_column(style="dim")
    for decision in route(question, limit=limit):
        because = "; ".join(decision.reasons) or "general-purpose fallback"
        table.add_row(decision.engine, f"{decision.score:.2f}", because)
    return table


def _plan_table(planner: Planner, question: str) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("engine", style="cyan", no_wrap=True)
    table.add_column("query")
    table.add_column("parameters", style="dim")

    for index, step in enumerate(planner.plan(question), start=1):
        extra = {k: v for k, v in step.params.items() if k not in {"q", "num"}}
        table.add_row(
            str(index),
            step.engine,
            f'"{step.query}"',
            " ".join(f"{k}={v}" for k, v in sorted(extra.items())) or "—",
        )
    return table


def _results_table(result: PlanResult) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("engine", style="cyan", no_wrap=True)
    table.add_column("results", justify="right")
    table.add_column("cost", no_wrap=True)
    table.add_column("note", style="dim")

    for step in result.steps:
        if step.error:
            cost, note = Text("—", style="dim"), Text(step.error[:60], style="red")
        elif step.from_cache:
            cost, note = Text("free", style="green"), Text("served from cache")
        else:
            cost, note = Text(f"{step.charged} search", style="yellow"), Text("")
        table.add_row(step.step.engine, str(step.evidence_count), cost, note)
    return table


def _evidence_panel(result: PlanResult, limit: int) -> Group:
    lines: list[Text | str] = []
    shown = 0

    for item in result.evidence:
        if shown >= limit:
            break
        shown += 1
        if isinstance(item, Series):
            values = [p.value for p in item.points]
            lines.append(Text(f"{shown}. {item.name}", style="bold magenta"))
            lines.append(Text(f"   {sparkline(values)}  {item.summarise()}", style="magenta"))
        elif isinstance(item, Document):
            lines.append(Text(f"{shown}. {item.title}", style="bold"))
            detail = item.source or ""
            if item.url:
                detail = f"{detail}  {item.url}" if detail else item.url
            if detail:
                lines.append(Text(f"   {detail}", style="dim"))
        lines.append("")

    remaining = len(result.evidence) - shown
    if remaining > 0:
        lines.append(Text(f"…and {remaining} more", style="dim italic"))
    return Group(*lines)


def _summary(result: PlanResult) -> Text:
    parts = Text()
    parts.append(f"{len(result.evidence)} unique results", style="bold")
    parts.append("   ")
    parts.append(f"{result.searches_charged} searches billed", style="yellow")
    if result.cache_hits:
        parts.append(f"   {result.cache_hits} from cache", style="green")
    if result.steps_skipped:
        parts.append(f"   {result.steps_skipped} steps skipped", style="green")
    parts.append(f"   ceiling {result.ceiling}", style="dim")
    return parts


def ask(
    question: str,
    *,
    budget: int,
    cache_dir: str,
    replay: bool,
    engines: int | None,
    show: int,
    console: Console,
) -> int:
    """Run a question and show the plan, the spend and the evidence."""
    mode = CacheMode.REPLAY if replay else CacheMode.AUTO
    cache = ResponseCache(cache_dir, mode=mode)

    with SerpApiClient(cache=cache) as client:
        planner = Planner(client, max_engines=engines or DEFAULT_MAX_ENGINES)
        locale = planner.locale_for(question)

        header = Text()
        header.append(question, style="bold white")
        header.append(f"\nbudget {budget} searches", style="dim")
        if locale is not None:
            header.append(f"   locale {locale.describe()}", style="dim")
        console.print(Panel(header, border_style="blue", padding=(0, 1)))

        console.print("\n[bold]Routing[/bold]")
        console.print(_routing_table(question, planner.max_engines))

        console.print("\n[bold]Plan[/bold]")
        console.print(_plan_table(planner, question))

        spend: list[int] = []

        def tick(*, spent: int, limit: int, engine: str | None) -> None:
            spend.append(spent)

        console.print()
        with console.status("[yellow]searching…", spinner="dots") as status:

            def watched(*, spent: int, limit: int, engine: str | None) -> None:
                tick(spent=spent, limit=limit, engine=engine)
                status.update(f"[yellow]{spent}/{limit} searches billed — {engine}")

            try:
                result = planner.run(question, budget=budget, listener=watched)
            except FrugalError as exc:
                console.print(f"[red]{type(exc).__name__}[/red]: {exc}")
                return 1

    console.print("[bold]Execution[/bold]")
    console.print(_results_table(result))

    console.print()
    console.print(_summary(result))
    console.print(Text(result.stopped_because, style="italic green"))

    for observation in result.observations:
        console.print(Text(f"  {observation.explain()}", style="dim"))

    if result.failures:
        console.print("\n[bold red]Failures[/bold red]")
        for failed in result.failures:
            console.print(Text(f"  {failed.step.engine}: {failed.error}", style="red"))

    if not result.evidence:
        # Every step failed. Returning success here would let a broken run look
        # like a question nobody has written about.
        console.print(
            Text("\nno evidence retrieved — every step failed", style="bold red")
        )
        return 1

    console.print("\n[bold]Evidence[/bold]")
    console.print(_evidence_panel(result, show))
    return 0


def plan_only(
    question: str, *, budget: int, cache_dir: str, engines: int | None, console: Console
) -> int:
    """Show what a question would cost, without issuing anything."""
    client = SerpApiClient(cache=ResponseCache(cache_dir, mode=CacheMode.REPLAY))
    planner = Planner(client, max_engines=engines or DEFAULT_MAX_ENGINES)
    dry = planner.dry_run(question, budget=budget)

    console.print(Panel(Text(question, style="bold white"), border_style="blue", padding=(0, 1)))
    console.print("\n[bold]Routing[/bold]")
    console.print(_routing_table(question, planner.max_engines))
    console.print("\n[bold]Plan[/bold]")
    console.print(_plan_table(planner, question))
    console.print()
    console.print(Text(dry.describe(), style="yellow"))
    console.print(Text("nothing was issued; no searches were spent", style="dim italic"))
    return 0


def doctor(console: Console) -> int:
    """Report on local setup. Returns a process exit code."""
    ok, detail = describe_key_source()

    console.print(f"frugal {__version__}   python {sys.version.split()[0]}\n")
    mark = "[green]ok[/green]" if ok else "[red]xx[/red]"
    console.print(f"[{mark}] SerpApi key: {detail}")

    cache = ResponseCache(DEFAULT_CACHE_DIR, mode=CacheMode.AUTO)
    entries = sum(1 for _ in cache.iter_entries()) if cache.directory.exists() else 0
    noun = "entry" if entries == 1 else "entries"
    console.print(f"[[green]ok[/green]] cache: {cache.directory} ({entries} recorded {noun})")

    if not ok:
        target = find_dotenv() or f"{DOTENV_NAME} (create it in the project root)"
        console.print("\n[bold]To fix:[/bold]")
        console.print("  1. Get a free key at https://serpapi.com/manage-api-key")
        console.print(f"  2. Put this line in {target}:")
        console.print(f"         {ENV_VAR}=your-actual-key")
        console.print(f"\n  {DOTENV_NAME} is gitignored, so the key will not be committed.")
        return 1

    console.print("\n[green]Setup looks good.[/green]")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frugal", description="A cost-aware search planner for SerpApi."
    )
    parser.add_argument("--version", action="version", version=f"frugal {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="check that the local setup is ready to run")

    for name, help_text in (
        ("ask", "run a question and show the plan, the spend and the evidence"),
        ("plan", "show what a question would cost, without issuing anything"),
    ):
        sub = subcommands.add_parser(name, help=help_text)
        sub.add_argument("question", help="the question to plan for")
        sub.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="search budget")
        sub.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="cache directory")
        sub.add_argument("--engines", type=int, default=None, help="engines to route across")
        if name == "ask":
            sub.add_argument(
                "--replay",
                action="store_true",
                help="serve only from cache; spends nothing and fails on a miss",
            )
            sub.add_argument("--show", type=int, default=8, help="results to display")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a command. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    console = Console()

    if args.command == "doctor":
        return doctor(console)
    if args.command == "plan":
        return plan_only(
            args.question,
            budget=args.budget,
            cache_dir=args.cache,
            engines=args.engines,
            console=console,
        )
    if args.command == "ask":
        return ask(
            args.question,
            budget=args.budget,
            cache_dir=args.cache,
            replay=args.replay,
            engines=args.engines,
            show=args.show,
            console=console,
        )
    return 1  # pragma: no cover - argparse rejects unknown commands first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
