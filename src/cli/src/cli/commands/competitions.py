import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.context import get as get_ctx
from competition.github_config import get_all_competitions, get_active_competitions

console = Console()


def competitions(
    all_: bool = typer.Option(False, "--all", "-a", help="Show all competitions, not just active"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh from source"),
):
    """Show competitions from the TPN config source."""
    ctx = get_ctx()

    current_block = 0
    try:
        from common.chain import get_subtensor
        current_block = get_subtensor(ctx.network).get_current_block()
    except Exception:
        pass

    if all_:
        specs = get_all_competitions(index_url=ctx.competition_url, force_refresh=refresh)
    else:
        specs = get_active_competitions(index_url=ctx.competition_url, current_block=current_block, force_refresh=refresh)

    if not specs:
        console.print("[yellow]No competitions found.[/yellow]")
        raise typer.Exit(0)

    for spec in specs:
        phase = spec.phase(current_block)
        remaining = spec.blocks_until_next_phase(current_block)
        minutes = remaining * 12 // 60

        lines = [
            f"[bold cyan]{spec.name}[/bold cyan]  [dim](ID: {spec.id})[/dim]",
        ]
        if spec.model_repo:
            lines.append(f"[dim]Base model:[/dim]     {spec.model_repo}")
        lines += [
            f"[dim]Phase:[/dim]          {phase.value}  ({remaining} blocks / ~{minutes} min remaining)",
            "",
            f"[dim]Start block:[/dim]    {spec.start_block}  ({spec.start_date or 'N/A'})",
            f"[dim]Commit ends:[/dim]    {spec.commit_end_block}  ({spec.commit_end_date or 'N/A'})",
            f"[dim]Scoring ends:[/dim]   {spec.scoring_end_block}  ({spec.scoring_end_date or 'N/A'})",
            "",
            "[bold]Benchmarks:[/bold]",
        ]
        for task in spec.benchmarks:
            lines.append(f"  {task.name:<14} min={task.min_score:.2f}  weight={task.weight}")

        lines += [
            "",
            f"[dim]Top N:[/dim]          {spec.top_n}",
            f"[dim]Emissions:[/dim]      {spec.emission_distribution}",
            f"[dim]Eval backend:[/dim]   {spec.eval.backend}",
        ]
        console.print(Panel("\n".join(lines), title="Competition", border_style="cyan"))
