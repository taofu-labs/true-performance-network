from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import load_competition_config
from cli.utils.context import get as get_ctx, resolve_competition_or_exit
from cli.utils.hf_auth import ensure_hf_auth

console = Console()


def publish(
    coldkey: str = typer.Option(..., "--wallet", "-w", help="Bittensor wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: str = typer.Option(..., "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
):
    """Make your HuggingFace repo public so validators can download your model."""
    ctx = get_ctx()
    resolve_competition_or_exit(ctx, competition_id)

    cfg = load_competition_config(coldkey, hotkey_name, competition_id)
    repo_id = cfg.get("repository")
    if not repo_id:
        console.print(
            f"[red]No repository found in config for {competition_id}.[/red]"
        )
        raise typer.Exit(1)

    api = ensure_hf_auth()

    console.print(f"[blue]Flipping {repo_id} to public...[/blue]")
    api.update_repo_settings(repo_id=repo_id, private=False)

    console.print(Panel(
        f"[green]✓ Repo is now public[/green]\n\n"
        f"[cyan]https://huggingface.co/{repo_id}[/cyan]\n\n"
        f"[dim]Validators will download and score your model during the scoring phase.[/dim]",
        title="Repo Published",
        border_style="green",
    ))
