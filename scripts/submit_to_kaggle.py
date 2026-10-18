"""Kaggle Code Competition submission automation script.

Handles:
1. Packaging the notebook and kernel-metadata.json
2. Pushing the notebook via `kaggle kernels push`
3. Monitoring notebook execution until 'complete'
4. Submitting the generated submission.parquet via:
   kaggle competitions submit -c name-that-llm -f submission.parquet -k <username>/<kernel> -m "Message"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_submission_notebook import build_submission_notebook
from name_that_llm.config import COMPETITION_NAME, SUBMISSION_FILE

console = Console()
KERNEL_DIR = PROJECT_ROOT / "kaggle_kernel"


def get_kaggle_username() -> str:
    """Detect Kaggle username from ~/.kaggle/kaggle.json or env, falling back to 'dustinober'."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                data = json.load(f)
                if data.get("username"):
                    return data["username"]
        except Exception:
            pass
    return os.environ.get("KAGGLE_USERNAME", "dustinober")


def prepare_kernel_bundle(
    kernel_slug: str,
    title: str = "Name That LLM - Solution Pipeline",
    enable_gpu: bool = False,
    username: str = "dustinober",
) -> Path:
    """Prepare the directory with notebook and kernel-metadata.json required by kaggle CLI."""
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ensure latest submission notebook is built
    build_submission_notebook()
    src_notebook = PROJECT_ROOT / "notebooks" / "kaggle_submission.ipynb"
    dest_notebook = KERNEL_DIR / "kaggle_submission.ipynb"
    shutil.copyfile(src_notebook, dest_notebook)

    # 2. Write kernel-metadata.json
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": "kaggle_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "false",  # Must be false for competition submission
        "dataset_sources": [],
        "competition_sources": [COMPETITION_NAME],
        "kernel_sources": [],
        "model_sources": [],
    }

    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[bold green]✓ Kernel bundle prepared in:[/bold green] {KERNEL_DIR}")
    return KERNEL_DIR


def check_auth() -> bool:
    """Verify Kaggle CLI credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "access_token").exists() or (kaggle_dir / "kaggle.json").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def run_command(cmd: list) -> subprocess.CompletedProcess:
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, capture_output=True, text=True)


def monitor_kernel(kernel_id: str, max_wait_seconds: int = 1800) -> bool:
    """Poll kernel status until complete or error."""
    start_time = time.time()
    console.print(f"[cyan]Waiting for notebook execution ({kernel_id}) to complete...[/cyan]")

    while time.time() - start_time < max_wait_seconds:
        res = run_command(["kaggle", "kernels", "status", kernel_id])
        output = (res.stdout + res.stderr).lower()

        if "complete" in output:
            console.print("[bold green]✓ Notebook execution completed successfully![/bold green]")
            return True
        elif "error" in output or "failed" in output:
            console.print(f"[bold red]✗ Notebook execution failed:[/bold red]\n{res.stdout}\n{res.stderr}")
            return False
        elif "running" in output:
            console.print("[dim]Notebook is currently running in Kaggle container...[/dim]")
        elif "queued" in output:
            console.print("[dim]Notebook is queued in Kaggle cluster...[/dim]")
        else:
            console.print(f"[dim]Status: {res.stdout.strip()}[/dim]")

        time.sleep(20)

    console.print("[bold red]Timeout reached while waiting for kernel run to complete.[/bold red]")
    return False


def main():
    parser = argparse.ArgumentParser(description="Push notebook to Kaggle and submit to competition.")
    parser.add_argument("-m", "--message", type=str, required=True, help="Submission message")
    parser.add_argument("-k", "--kernel-slug", type=str, default="name-that-llm-baseline", help="Kernel slug")
    parser.add_argument("-u", "--username", type=str, default=None, help="Kaggle username")
    parser.add_argument("-v", "--version", type=str, default=None, help="Specific kernel version to submit")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU on Kaggle")
    parser.add_argument("--push-only", action="store_true", help="Only push notebook without submitting")
    parser.add_argument("--submit-only", action="store_true", help="Submit already-executed kernel output")
    args = parser.parse_args()

    username = args.username or get_kaggle_username()
    kernel_id = f"{username}/{args.kernel_slug}"

    console.rule(f"[bold cyan]Kaggle Competition Submission: {kernel_id}[/bold cyan]")

    if not check_auth():
        console.print("[bold red]Kaggle credentials not found in ~/.kaggle/kaggle.json![/bold red]")
        console.print(
            "\n[bold yellow]Setup Instructions:[/bold yellow]\n"
            "1. Visit https://www.kaggle.com/settings -> API -> 'Create New Token'\n"
            "2. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
        )
        sys.exit(1)

    if not args.submit_only:
        # 1. Prepare bundle
        prepare_kernel_bundle(
            kernel_slug=args.kernel_slug,
            enable_gpu=args.gpu,
            username=username,
        )

        # 2. Push notebook to Kaggle
        console.print(f"[bold cyan]Pushing notebook to Kaggle ({kernel_id})...[/bold cyan]")
        push_res = run_command(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])
        console.print(push_res.stdout)
        if push_res.returncode != 0:
            console.print(f"[bold red]Failed to push kernel:[/bold red] {push_res.stderr}")
            sys.exit(1)

        if args.push_only:
            console.print(f"[bold green]✓ Successfully pushed to Kaggle:[/bold green] https://www.kaggle.com/code/{kernel_id}")
            return

        # 3. Monitor execution
        success = monitor_kernel(kernel_id)
        if not success:
            sys.exit(1)

    # 4. Submit to competition
    console.print(f"[bold cyan]Submitting {SUBMISSION_FILE} from kernel {kernel_id} to {COMPETITION_NAME}...[/bold cyan]")
    submit_cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION_NAME,
        "-f",
        SUBMISSION_FILE,
        "-k",
        kernel_id,
        "-m",
        args.message,
    ]
    version_str = str(args.version) if args.version else "1"
    submit_cmd.extend(["-v", version_str])

    sub_res = run_command(submit_cmd)
    console.print(sub_res.stdout)
    if sub_res.returncode != 0:
        console.print(f"[bold red]Submission command failed:[/bold red] {sub_res.stderr}")
        sys.exit(1)

    console.print(f"[bold green]✓ Successfully submitted to {COMPETITION_NAME}![/bold green]")
    console.print(f"Check your leaderboard standing at: [cyan]https://www.kaggle.com/competitions/{COMPETITION_NAME}/submissions[/cyan]")


if __name__ == "__main__":
    main()
