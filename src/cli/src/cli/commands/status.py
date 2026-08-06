from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import load_competition_config, identity_dir
from cli.utils.context import get as get_ctx
from competition.leader_config_client import get_active_competitions, is_competition

console = Console()


def status(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: Optional[str] = typer.Option(None, "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
):
    """Show your current submission status."""
    ctx = get_ctx()

    if competition_id is not None and not is_competition(ctx.leader_url, competition_id):
        console.print(f"[red]Competition '{competition_id}' not found.[/red]")
        raise typer.Exit(1)

    current_block = 0
    try:
        from common.chain import current_block as get_current_block, get_subtensor
        current_block = get_current_block(get_subtensor(ctx.network))
    except Exception:
        pass

    active_specs = get_active_competitions(base_url=ctx.leader_url, current_block=current_block)
    lines = [
        f"[dim]Wallet:[/dim]        {coldkey} / {hotkey_name}",
        f"[dim]Current block:[/dim] {current_block}",
    ]

    for spec in active_specs:
        phase = spec.phase(current_block)
        remaining = spec.blocks_until_next_phase(current_block)
        lines += [
            f"[dim]Competition:[/dim]   {spec.name} ([dim]{spec.id}[/dim])",
            f"[dim]Phase:[/dim]         {phase.value}  ({remaining} blocks remaining)",
        ]

    lines.append("")

    # Determine which competition config(s) to show
    target_id = competition_id or (active_specs[0].id if active_specs else None)
    if target_id:
        cfg = load_competition_config(coldkey, hotkey_name, target_id)
        if cfg:
            claims_display = ", ".join(f"{c['b']}:{c['s']}" for c in (cfg.get("claims") or []))
            lines += [
                f"[bold]Pending commit ({target_id}):[/bold]",
                f"  Repo:         [cyan]{cfg.get('repository', '[none]')}[/cyan]",
                f"  File:         [dim]{cfg.get('file', '[none]')}[/dim]",
                f"  Size:         {cfg.get('file_size', 0):,} bytes",
                f"  Claims:       {claims_display or '[none]'}",
            ]
            if cfg.get("commit_end_block"):
                lines.append(f"  Reveals at:   block {cfg['commit_end_block']}")
        else:
            lines += [
                f"[yellow]No config found for competition {target_id}.[/yellow]",
                f"[dim]Config path: {identity_dir(coldkey, hotkey_name) / (target_id + '.json')}[/dim]",
            ]
    else:
        lines += [
            "[yellow]No active competition found.[/yellow]",
            "Pass [green]--competition <id>[/green] to see a specific competition's state.",
        ]

    console.print(Panel("\n".join(lines), title="TPN Status", border_style="cyan"))
