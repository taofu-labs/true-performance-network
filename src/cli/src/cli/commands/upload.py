"""
Upload a GGUF model to a private HuggingFace repo.
Step 1 of the miner workflow — done before commit.
"""
import hashlib
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from cli.utils.config import load_competition_config, save_competition_config
from cli.utils.context import get as get_ctx, resolve_competition_or_exit
from cli.utils.hf_auth import ensure_hf_auth

console = Console()


def upload(
    gguf_path: Path = typer.Argument(..., help="Path to your .gguf file"),
    repo_name: str = typer.Option(..., "--repo", "-r", help="HF repo name (e.g. my-model-q4)"),
    coldkey: str = typer.Option(..., "--coldkey", "-w", help="Bittensor coldkey name"),
    hotkey_name: str = typer.Option("default", "--hotkey", help="Hotkey name"),
    competition_id: str = typer.Option(..., "--competition", "-c", help="Competition ID (e.g. tpn-001)"),
):
    """Upload your GGUF model to a private HuggingFace repo."""
    ctx = get_ctx()
    resolve_competition_or_exit(ctx, competition_id)

    if not gguf_path.exists():
        console.print(f"[red]File not found: {gguf_path}[/red]")
        raise typer.Exit(1)

    api = ensure_hf_auth()
    username = api.whoami()["name"]
    repo_id = f"{username}/{repo_name}"

    console.print(f"[blue]Creating private repo: {repo_id}[/blue]")
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)

    console.print(f"[blue]Uploading {gguf_path.name} ({gguf_path.stat().st_size:,} bytes)...[/blue]")
    commit_info = api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
    )
    revision = commit_info.oid  # immutable commit SHA of this upload

    file_sha256 = _sha256(gguf_path)

    cfg = load_competition_config(coldkey, hotkey_name, competition_id)
    cfg.update({
        "repository": repo_id,
        "file": gguf_path.name,
        "file_sha256": file_sha256,
        "huggingface_revision": revision,
    })
    save_competition_config(coldkey, hotkey_name, competition_id, cfg)

    console.print(Panel(
        f"[green]✓ Uploaded successfully[/green]\n\n"
        f"Repo:    [cyan]{repo_id}[/cyan]\n"
        f"File:    [dim]{gguf_path.name}[/dim]\n"
        f"Rev:     [dim]{revision}[/dim]\n"
        f"SHA256:  [dim]{file_sha256}[/dim]\n\n"
        f"[dim]Next step: run `tpn commit --wallet {coldkey} --hotkey {hotkey_name} --competition {competition_id}`[/dim]",
        title="Upload Complete",
        border_style="green",
    ))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
