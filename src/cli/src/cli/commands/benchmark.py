"""
Run a competition's benchmarks against an uploaded model.

Miners benchmark their own models before committing. This drives the benchmark
service's billing API with an API key, then saves the resulting coordinator run
ids into the competition config so `tpn commit` can put them on chain.
"""
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from cli.utils.config import load_competition_config, save_competition_config
from cli.utils.context import get as get_ctx, resolve_competition_or_exit
from common.urls import InvalidBaseUrl
from competition.billing_client import (
    DEFAULT_BILLING_URL,
    BillingClient,
    BillingError,
    build_request,
)

console = Console()


def benchmark(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: str = typer.Option(..., "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="Benchmark service API key (bapi_...). Defaults to $BENCHMARK_API_KEY.",
        envvar="BENCHMARK_API_KEY",
    ),
    billing_url: str = typer.Option(
        DEFAULT_BILLING_URL, "--billing-url",
        help="Benchmark service billing API base URL",
        envvar="BENCHMARK_BILLING_URL",
    ),
    only: Optional[str] = typer.Option(
        None, "--only",
        help="Comma-separated benchmark names to run (default: all the competition requires)",
    ),
    rerun: bool = typer.Option(
        False, "--rerun",
        help="Re-run benchmarks that already have a saved run id",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the price confirmation prompt"),
    force_price: bool = typer.Option(
        False, "--force-price",
        help="Accept the current price without requesting a quote first",
    ),
):
    """Run this competition's benchmarks and save the resulting run ids."""
    ctx = get_ctx()
    spec = resolve_competition_or_exit(ctx, competition_id)

    if not api_key:
        console.print(
            "[red]No benchmark API key.[/red]\n"
            "[dim]Pass --api-key, or set BENCHMARK_API_KEY. Create a key on the "
            "benchmark service's Tokens page, and add credits there first — "
            "API keys cannot buy credits.[/dim]"
        )
        raise typer.Exit(1)

    cfg = load_competition_config(coldkey, hotkey_name, competition_id)
    missing = [f for f in ("repository", "file", "huggingface_revision") if not cfg.get(f)]
    if missing:
        console.print(
            f"[red]Missing upload data: {', '.join(missing)}[/red]\n"
            f"[dim]Run `tpn upload --wallet {coldkey} --hotkey {hotkey_name} "
            f"--competition {competition_id} ...` first.[/dim]"
        )
        raise typer.Exit(1)

    wanted = [t.name for t in spec.benchmarks]
    if only:
        requested = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in requested if n not in wanted]
        if unknown:
            console.print(
                f"[red]Not part of {competition_id}: {', '.join(unknown)}[/red]\n"
                f"[dim]This competition requires: {', '.join(wanted)}[/dim]"
            )
            raise typer.Exit(1)
        wanted = requested

    existing = {r["b"]: r["r"] for r in (cfg.get("runs") or [])}
    todo = wanted if rerun else [n for n in wanted if n not in existing]
    if not todo:
        console.print(
            f"[green]All benchmarks already have run ids.[/green] "
            f"[dim]Use --rerun to run them again.[/dim]"
        )
        _print_runs(existing, wanted)
        raise typer.Exit(0)

    try:
        client = BillingClient(base_url=billing_url, api_key=api_key)
    except (BillingError, InvalidBaseUrl) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]Benchmarking[/bold] [cyan]{cfg['repository']}[/cyan]"
        f" [dim]@{cfg['huggingface_revision'][:12]}/{cfg['file']}[/dim]\n"
        f"[dim]Benchmarks: {', '.join(todo)}[/dim]\n"
    )

    # Quote everything first so the miner sees the full bill before any spend.
    quotes = {}
    if not force_price:
        total_cents = 0
        for name in todo:
            request = build_request(cfg["repository"], cfg["huggingface_revision"], cfg["file"], name)
            try:
                quotes[name] = client.quote(request)
            except BillingError as e:
                console.print(f"[red]Quote failed for {name}: {e}[/red]")
                raise typer.Exit(1)
            total_cents += quotes[name].payable_cents

        table = Table(show_header=True, header_style="bold")
        table.add_column("Benchmark")
        table.add_column("Price", justify="right")
        table.add_column("Notes")
        for name in todo:
            q = quotes[name]
            notes = []
            if q.reuse_eligible:
                notes.append("cached result")
            if q.review_required:
                notes.append("needs review")
            table.add_row(name, f"${q.charged_cents / 100:,.2f}", ", ".join(notes) or "—")
        console.print(table)

        credit = next(iter(quotes.values())).available_credit_cents
        console.print(
            f"\n[bold]Total:[/bold] ${total_cents / 100:,.2f}  "
            f"[dim]Available credit: ${credit / 100:,.2f}[/dim]"
        )
        if total_cents > credit:
            console.print(
                f"[red]Not enough credit — ${(total_cents - credit) / 100:,.2f} short.[/red]\n"
                "[dim]Top up on the benchmark service's web GUI; API keys cannot buy credits.[/dim]"
            )
            raise typer.Exit(1)

        if not yes and not typer.confirm(f"Spend ${total_cents / 100:,.2f}?"):
            raise typer.Exit(0)

    # Submit each benchmark, then wait only until the coordinator assigns a run
    # id — not for the benchmark to finish. The id is what goes on chain.
    runs = dict(existing)
    failures = []
    for name in todo:
        request = build_request(cfg["repository"], cfg["huggingface_revision"], cfg["file"], name)
        try:
            order = client.submit(request, quotes.get(name))
            console.print(f"[dim]{name}: order {order.id} ({order.status}) — waiting for run id[/dim]")
            order = client.wait_for_run_id(order.id)
        except BillingError as e:
            console.print(f"[red]{name} failed: {e}[/red]")
            failures.append(name)
            continue

        runs[name] = order.commit_id
        console.print(f"[green]✓[/green] {name} → [cyan]{order.commit_id}[/cyan]")

        # Persist after every success: a later failure must not lose the run
        # ids already paid for.
        cfg["runs"] = [{"b": b, "r": r} for b, r in runs.items()]
        save_competition_config(coldkey, hotkey_name, competition_id, cfg)

    console.print()
    _print_runs(runs, wanted)

    if failures:
        console.print(f"\n[red]Failed: {', '.join(failures)}[/red] [dim]Re-run to retry.[/dim]")
        raise typer.Exit(1)

    console.print(
        "\n[dim]Benchmarks are running. Results are not needed to commit — the "
        "run ids are.[/dim]\n"
        f"[dim]Commit with: tpn commit -w {coldkey} -c {competition_id}[/dim]"
    )


def _print_runs(runs: dict, wanted: list) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Benchmark")
    table.add_column("Run id")
    for name in wanted:
        table.add_row(name, runs.get(name) or "[dim]—[/dim]")
    console.print(table)
