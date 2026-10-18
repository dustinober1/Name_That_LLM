"""Script to download the official Name That LLM dataset from Kaggle."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from rich.console import Console

from name_that_llm.config import COMPETITION_NAME, RAW_DATA_DIR

console = Console()


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist in environment or default file path."""
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "access_token").exists() or (kaggle_dir / "kaggle.json").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_competition_data() -> None:
    """Download competition data into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    train_file = RAW_DATA_DIR / "train.csv"
    if train_file.exists():
        console.print(f"[bold green]✓ Competition data already exists at:[/bold green] {RAW_DATA_DIR}")
        return

    if not check_kaggle_credentials():
        console.print("[bold red]Error: Kaggle API credentials not found![/bold red]")
        console.print(
            "\n[bold yellow]To set up your Kaggle credentials:[/bold yellow]\n"
            "1. Visit [cyan]https://www.kaggle.com/settings[/cyan]\n"
            "2. Scroll down to the [bold]API[/bold] section and click [bold]'Create New Token'[/bold]\n"
            "   (This downloads a file named `kaggle.json`)\n"
            "3. Run the following shell command to place it in your ~/.kaggle folder:\n\n"
            "   [bold green]mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json[/bold green]\n\n"
            "4. Make sure you have accepted the competition rules on the Kaggle website:\n"
            f"   [cyan]https://www.kaggle.com/competitions/{COMPETITION_NAME}/data[/cyan]\n"
            "5. Re-run this script: [bold cyan]python scripts/download_data.py[/bold cyan]\n"
        )
        sys.exit(1)

    console.print(f"[bold cyan]Downloading dataset for '{COMPETITION_NAME}' to {RAW_DATA_DIR}...[/bold cyan]")
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION_NAME,
        "-p",
        str(RAW_DATA_DIR),
    ]

    try:
        subprocess.run(cmd, check=True)
        # Check for downloaded zip
        zip_path = RAW_DATA_DIR / f"{COMPETITION_NAME}.zip"
        if zip_path.exists():
            console.print("[dim]Unzipping dataset...[/dim]")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            zip_path.unlink()  # Remove zip after extraction
        console.print(f"[bold green]✓ Dataset successfully extracted to {RAW_DATA_DIR}![/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to download dataset:[/bold red] {e}")
        console.print(
            "Please ensure you have accepted the competition rules at:\n"
            f"https://www.kaggle.com/competitions/{COMPETITION_NAME}/rules"
        )
        sys.exit(1)


if __name__ == "__main__":
    download_competition_data()
