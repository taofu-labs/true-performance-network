import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.context import get as get_ctx, current_block_safe
from competition.leader_config_client import get_all_competitions, get_active_competitions

console = Console()


def competitions(
    all_: bool = typer.Option(False, "--all", "-a", help="Show all competitions, not just active"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh from source"),
):
    """Show competitions from the TPN config source."""
    ctx = get_ctx()
    current_block = current_block_safe(ctx)

    if all_:
        specs = get_all_competitions(base_url=ctx.leader_url, force_refresh=refresh)
    else:
        specs = get_active_competitions(base_url=ctx.leader_url, current_block=current_block, force_refresh=refresh)

    if not specs:
        console.print("[yellow]No competitions found.[/yellow]")
        raise typer.Exit(0)

    for spec in specs:
        phase = spec.phase(current_block)
        remaining = spec.blocks_until_next_phase(current_block)
        minutes = remaining * ctx.block_time // 60

        lines = [
            f"[bold cyan]{spec.name}[/bold cyan]  [dim](ID: {spec.id})[/dim]",
        ]
        if spec.model_repo:
            lines.append(f"[dim]Base model:[/dim]     {spec.model_repo}")
        lines += [
            f"[dim]Type:[/dim]           {spec.competition_type.value}",
            f"[dim]Phase:[/dim]          {phase.value}  ({remaining} blocks / ~{minutes} min remaining)",
            "",
            f"[dim]Start block:[/dim]    {spec.start_block}",
            f"[dim]Commit ends:[/dim]    {spec.commit_end_block}",
            f"[dim]Scoring ends:[/dim]   {spec.scoring_end_block}",
            "",
            "[bold]Benchmarks:[/bold]",
        ]
        for task in spec.benchmarks:
            lines.append(f"  {task.name:<14} min={task.min_score:.2f}  weight={task.weight}")
        if spec.max_memory_kb:
            lines.append(f"\n[dim]Max memory:[/dim]     {spec.max_memory_kb:,} KB")

        lines += [
            "",
            f"[dim]Top N:[/dim]          {spec.top_n}",
            f"[dim]Emissions:[/dim]      {spec.emission_distribution}",
        ]
        console.print(Panel("\n".join(lines), title="Competition", border_style="cyan"))
