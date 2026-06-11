"""
Submit a TimeLocked Commit to chain.
The payload is TLE-encrypted and auto-decrypted by the chain at commit_end_block.
"""
import json
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import load_competition_config, save_competition_config
from cli.utils.context import get as get_ctx
from common.chain import timelocked_commit, is_hotkey_registered, get_subtensor, get_wallet
from common.models.submission import Claim, build_reveal_payload
from competition.github_config import get_competition_by_id

console = Console()


def commit(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: str = typer.Option(..., "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
    claims: Optional[str] = typer.Option(None, "--claims", help='JSON claims e.g. \'[{"b":"MMLU","s":0.93}]\''),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Override config file (JSON)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Prepare commit but do not write to chain"),
):
    """Submit a TimeLocked Commit for a competition."""
    ctx = get_ctx()

    wallet = get_wallet(coldkey=coldkey, hotkey=hotkey_name, wallet_path=ctx.wallet_path)
    hotkey_ss58 = wallet.hotkey.ss58_address
    coldkey_ss58 = wallet.coldkey.ss58_address

    subtensor = get_subtensor(ctx.network)
    current_block = subtensor.get_current_block()

    # ── Registration ─────────────────────────────────────────────────
    if not is_hotkey_registered(subtensor, hotkey_ss58, ctx.netuid):
        console.print(
            f"[red]Hotkey {hotkey_ss58[:12]}... is not registered on subnet {ctx.netuid}.[/red]\n"
            "[dim]Run `tpn register` first.[/dim]"
        )
        raise typer.Exit(1)

    # ── Competition valid + OPEN ─────────────────────────────────────
    spec = get_competition_by_id(index_url=ctx.competition_url, competition_id=competition_id)
    if spec is None:
        console.print(f"[red]Competition '{competition_id}' not found.[/red]")
        raise typer.Exit(1)

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
    missing = [f for f in ("repository", "file", "file_sha256", "file_size") if not cfg.get(f)]
    if missing:
        console.print(
            f"[red]Missing upload data: {', '.join(missing)}[/red]\n"
            f"[dim]Run `tpn upload --wallet {coldkey} --hotkey {hotkey_name} --competition {competition_id} ...` first.[/dim]"
        )
        raise typer.Exit(1)

    # ── Claims — resolve, validate, prompt if needed ─────────────────
    parsed_claims = _resolve_claims(claims, cfg, spec)
    if parsed_claims is None:
        raise typer.Exit(1)

    # ── Summary + confirmation ────────────────────────────────────────────────
    claims_display = ", ".join(f"{c.b}:{c.s}" for c in parsed_claims)
    console.print(Panel(
        f"[bold]Submission summary[/bold]\n\n"
        f"Wallet:           [cyan]{coldkey}[/cyan] / [cyan]{hotkey_name}[/cyan]\n"
        f"Competition:      [cyan]{competition_id}[/cyan] — {spec.name}\n"
        f"Repo:             [cyan]{cfg['repository']}[/cyan]\n"
        f"File:             [dim]{cfg['file']}[/dim]\n"
        f"SHA256:           [dim]{cfg['file_sha256'][:24]}...[/dim]\n"
        f"Size:             [dim]{cfg['file_size']:,} bytes[/dim]\n"
        f"Claims:           [dim]{claims_display}[/dim]\n"
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
        file_size=cfg["file_size"],
        claims=parsed_claims,
    )

    # ── Persist state ─────────────────────────────────────────────────────────
    cfg["claims"] = [{"b": c.b, "s": c.s} for c in parsed_claims]
    cfg["commit_end_block"] = spec.commit_end_block
    save_competition_config(coldkey, hotkey_name, competition_id, cfg)

    # ── Submit ────────────────────────────────────────────────────────────────
    if not dry_run:
        blocks_until_reveal = max(1, spec.commit_end_block - current_block)
        success = timelocked_commit(
            subtensor=subtensor,
            wallet=wallet,
            netuid=ctx.netuid,
            reveal_payload=reveal_payload,
            blocks_until_reveal=blocks_until_reveal,
            block_time=ctx.block_time,
        )
        status = "[green]✓ TimeLocked Commit submitted[/green]" if success else "[red]✗ Chain write failed[/red]"
    else:
        status = "[yellow]Dry run — not written to chain[/yellow]"

    console.print(status)


def _resolve_claims(
    claims_flag: Optional[str],
    cfg: dict,
    spec,
) -> Optional[list]:
    """
    Resolve claims from (in priority order):
      1. cfg['claims'] field
      2. --claims CLI flag
      3. Interactive prompt per benchmark

    Returns list[Claim] or None on validation error.
    """
    benchmark_names = [t.name for t in spec.benchmarks]

    # Try config file first, then CLI flag
    raw_claims = cfg.get("claims") or None
    if raw_claims is None and claims_flag is not None:
        try:
            raw_claims = json.loads(claims_flag)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON for --claims.[/red]")
            return None

    if raw_claims is not None:
        # Validate structure
        if not isinstance(raw_claims, list) or not all(
            isinstance(c, dict) and isinstance(c.get("b"), str)
            for c in raw_claims
        ):
            console.print('[red]--claims must be a JSON array of {"b": str, "s": number} objects.[/red]')
            return None
        provided = {c["b"] for c in raw_claims}
        missing_keys = [n for n in benchmark_names if n not in provided]
        if missing_keys:
            console.print(f"[yellow]Claims missing benchmarks: {', '.join(missing_keys)} — prompting.[/yellow]")
            extra = _prompt_claims(missing_keys)
            raw_claims = list(raw_claims) + extra
    else:
        # No claims anywhere — prompt all
        console.print(f"[yellow]No claims found. Enter scores for each benchmark (0–1 scale).[/yellow]")
        raw_claims = _prompt_claims(benchmark_names)

    return [Claim(b=c["b"], s=float(c.get("s") or 0.0)) for c in raw_claims]


def _prompt_claims(benchmark_names: list) -> list:
    """Prompt user for a score per benchmark name."""
    result = []
    for name in benchmark_names:
        score = typer.prompt(f"  Score for {name} (0–1)", type=float)
        result.append({"b": name, "s": max(0.0, min(1.0, score))})
    return result
