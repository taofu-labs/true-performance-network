import typer
from rich.console import Console
from huggingface_hub import HfApi, login

console = Console()


def ensure_hf_auth() -> HfApi:
    """Return authenticated HfApi. Prompts for token if not logged in."""
    api = HfApi()
    try:
        api.whoami()
        return api
    except Exception:
        pass

    console.print("[yellow]Not logged in to HuggingFace.[/yellow]")
    console.print("[dim]Get a write token at: https://huggingface.co/settings/tokens[/dim]")
    token = typer.prompt("HuggingFace token (input will not be visible)", hide_input=True)
    try:
        login(token=token, add_to_git_credential=False)
        api = HfApi()
        api.whoami()
        return api
    except Exception:
        console.print("[red]Invalid token. Get a valid write token at: https://huggingface.co/settings/tokens[/red]")
        raise typer.Exit(1)
