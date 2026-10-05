"""
Submit a TimeLocked Commit to chain.
The payload is TLE-encrypted and auto-decrypted by the chain at commit_end_block.
"""
import json
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from pydantic import ValidationError
from rich.panel import Panel
from cli.utils.config import load_competition_config, save_competition_config
from cli.utils.context import get as get_ctx, resolve_competition_or_exit
from common.chain import current_block as get_current_block, timelocked_commit, is_hotkey_registered, get_subtensor, get_wallet
from common.models.submission import BenchmarkRun, build_reveal_payload

console = Console()


def commit(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: str = typer.Option(..., "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
    runs: Optional[str] = typer.Option(None, "--runs", help='JSON benchmark runs e.g. \'[{"b":"mmlu","r":"r1234"}]\''),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Override config file (JSON)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Prepare commit but do not write to chain"),
):
    """Submit a TimeLocked Commit for a competition."""
    ctx = get_ctx()

    wallet = get_wallet(coldkey=coldkey, hotkey=hotkey_name, wallet_path=ctx.wallet_path)
    hotkey_ss58 = wallet.hotkey.ss58_address
    coldkey_ss58 = wallet.coldkey.ss58_address

    subtensor = get_subtensor(ctx.network)
    current_block = get_current_block(subtensor)

    # ── Registration ─────────────────────────────────────────────────
    if not is_hotkey_registered(subtensor, hotkey_ss58, ctx.netuid):
        console.print(
            f"[red]Hotkey {hotkey_ss58[:12]}... is not registered on subnet {ctx.netuid}.[/red]\n"
            "[dim]Run `tpn register` first.[/dim]"
        )
        raise typer.Exit(1)

    # ── Competition valid + OPEN ─────────────────────────────────────
    spec = resolve_competition_or_exit(ctx, competition_id)

    from common.models.competition import CompetitionPhase
    phase = spec.phase(current_block)
    if phase != CompetitionPhase.OPEN:
        console.print(
            f"[red]Competition '{competition_id}' is not in OPEN phase (current: {phase.value}).[/red]"
        )
        raise typer.Exit(1)

    # ── Load config ───────────────────────────────────────────────────────────
    if config_file is not None:
        if not config_file.exists():
            console.print(f"[red]Config file not found: {config_file}[/red]")
            raise typer.Exit(1)
        cfg = json.loads(config_file.read_text())
    else:
        cfg = load_competition_config(coldkey, hotkey_name, competition_id)

    # ── Validate upload fields ────────────────────────────────────────────────
    missing = [f for f in ("repository", "file", "file_sha256", "huggingface_revision") if not cfg.get(f)]
    if missing:
        console.print(
            f"[red]Missing upload data: {', '.join(missing)}[/red]\n"
            f"[dim]Run `tpn upload --wallet {coldkey} --hotkey {hotkey_name} --competition {competition_id} ...` first.[/dim]"
        )
        raise typer.Exit(1)

    # ── Benchmark runs — resolve, validate, prompt if needed ─────────
    parsed_runs = _resolve_runs(runs, cfg, spec)
    if parsed_runs is None:
        raise typer.Exit(1)

    # ── Max memory — prompt if not already in config ──────────────────
    max_memory = cfg.get("max_memory")
    if not max_memory:
        max_memory = typer.prompt("Max memory usage during inference (KB)", type=int)
        while max_memory <= 0:
            console.print("[red]max_memory must be > 0[/red]")
            max_memory = typer.prompt("Max memory usage during inference (KB)", type=int)

    # ── Summary + confirmation ────────────────────────────────────────────────
    runs_display = ", ".join(f"{r.b}:{r.r}" for r in parsed_runs)
    console.print(Panel(
        f"[bold]Submission summary[/bold]\n\n"
        f"Wallet:           [cyan]{coldkey}[/cyan] / [cyan]{hotkey_name}[/cyan]\n"
        f"Competition:      [cyan]{competition_id}[/cyan] — {spec.name}\n"
        f"Repo:             [cyan]{cfg['repository']}[/cyan]\n"
        f"File:             [dim]{cfg['file']}[/dim]\n"
        f"Rev:              [dim]{cfg['huggingface_revision']}[/dim]\n"
        f"SHA256:           [dim]{cfg['file_sha256'][:24]}...[/dim]\n"
        f"Max memory:       [dim]{max_memory:,} KB[/dim]\n"
        f"Benchmark runs:   [dim]{runs_display}[/dim]\n"
        f"Auto-reveal at:   block [cyan]{spec.commit_end_block}[/cyan]",
        title="TimeLocked Commit",
        border_style="cyan",
    ))

    if not dry_run and not typer.confirm("Submit commit?"):
        raise typer.Exit(0)

    # ── Build payload ─────────────────────────────────────────────────────────
    reveal_payload = build_reveal_payload(
        competition_id=spec.id,
        repository=cfg["repository"],
        file=cfg["file"],
        file_sha256=cfg["file_sha256"],
        max_memory=max_memory,
        runs=parsed_runs,
        huggingface_revision=cfg["huggingface_revision"],
    )

    # previous_reveal_round of the last commit for THIS competition, so a
    # republish (updating the submission before it reveals) replaces that
    # field on-chain instead of adding a second pending field for the same
    # competition.
    previous_reveal_round = cfg.get("reveal_round")

    # ── Persist state ─────────────────────────────────────────────────────────
    cfg["runs"] = [{"b": r.b, "r": r.r} for r in parsed_runs]
    cfg["max_memory"] = max_memory
    cfg["commit_end_block"] = spec.commit_end_block
    save_competition_config(coldkey, hotkey_name, competition_id, cfg)

    # ── Submit ────────────────────────────────────────────────────────────────
    if not dry_run:
        current_block = get_current_block(subtensor)
        blocks_until_reveal = max(1, spec.commit_end_block - current_block)
        result = timelocked_commit(
            subtensor=subtensor,
            wallet=wallet,
            netuid=ctx.netuid,
            reveal_payload=reveal_payload,
            blocks_until_reveal=blocks_until_reveal,
            block_time=ctx.block_time,
            previous_reveal_round=previous_reveal_round,
        )
        if result.success:
            cfg["reveal_round"] = result.reveal_round
            save_competition_config(coldkey, hotkey_name, competition_id, cfg)
        status = "[green]✓ TimeLocked Commit submitted[/green]" if result.success else "[red]✗ Chain write failed[/red]"
    else:
        status = "[yellow]Dry run — not written to chain[/yellow]"

    console.print(status)


def _resolve_runs(
    runs_flag: Optional[str],
    cfg: dict,
    spec,
) -> Optional[list]:
    """
    Resolve benchmark run ids from (in priority order):
      1. cfg['runs'] field
      2. --runs CLI flag
      3. Interactive prompt per benchmark

    Returns list[BenchmarkRun] or None on validation error.
    """
    benchmark_names = [t.name for t in spec.benchmarks]

    # Try config file first, then CLI flag
    raw_runs = cfg.get("runs") or None
    if raw_runs is None and runs_flag is not None:
        try:
            raw_runs = json.loads(runs_flag)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON for --runs.[/red]")
            return None

    if raw_runs is not None:
        # Validate structure
        if not isinstance(raw_runs, list) or not all(
            isinstance(r, dict) and isinstance(r.get("b"), str) and isinstance(r.get("r"), str)
            for r in raw_runs
        ):
            console.print('[red]--runs must be a JSON array of {"b": str, "r": str} objects.[/red]')
            return None
        provided = {r["b"] for r in raw_runs}
        missing_keys = [n for n in benchmark_names if n not in provided]
        if missing_keys:
            console.print(f"[yellow]Runs missing benchmarks: {', '.join(missing_keys)} — prompting.[/yellow]")
            extra = _prompt_runs(missing_keys)
            raw_runs = list(raw_runs) + extra
    else:
        # No runs anywhere — prompt all
        console.print("[yellow]No benchmark runs found. Enter the coordinator run id for each benchmark.[/yellow]")
        raw_runs = _prompt_runs(benchmark_names)

    # Catch a mistyped run id here rather than letting it silently score 0.0
    # at reveal time, when it can no longer be corrected.
    try:
        return [BenchmarkRun(b=r["b"], r=str(r["r"]).strip()) for r in raw_runs]
    except ValidationError as e:
        console.print(f"[red]Invalid run id: {e.errors()[0]['msg']}[/red]")
        return None


def _prompt_runs(benchmark_names: list) -> list:
    """Prompt user for a coordinator run id per benchmark name."""
    result = []
    for name in benchmark_names:
        run_id = typer.prompt(f"  Run id for {name}")
        result.append({"b": name, "r": run_id.strip()})
    return result
