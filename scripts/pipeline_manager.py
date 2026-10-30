"""Automated Kaggle 5-Submission Pipeline Manager (v35 - v39).

Manages the 5-experiment SOTA queue:
- v35: Verified SOTA 0.663 Pipeline (Dual-T4 Data Parallel + ATLAS 20% + 15% Comp Stack + 34-Marker Bayesian Typography + Robust Llama)
- v36: 0.663 Pipeline + V40A Family-Only Stylometry Sibling Residual (5-Fold GroupKFold OOF Calibrated, Cross-Family Invariant)
- v37: NTL v41 Grandmaster Horizon (Calibrated Pool Stylometry 0.50 + Expanded ATLAS 25% Margin 0.35 + Soft Sinkhorn 0.50/0.40)
- v38: Grandmaster Horizon v41 + V40A Sibling Style Residual (max_alpha=0.06) + Zero-Violation Typography Guards
- v39: Grand Master Consensus Blend (Balanced Pool 0.42 + ATLAS 22% Margin 0.33 + Per-Row Member Normalization + Sibling Residual + Sinkhorn 0.48)
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_QUEUE = [
    {
        "id": "v35",
        "slug": "name-that-llm-sota-0663-v35",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v35_sota_0663",
        "version": "2",
        "message": "Verified SOTA 0.663 Reproduction (Dual-T4 Data Parallel + ATLAS 20% + 15% Comp Stack + 34-Marker Bayesian Typography + Robust Llama) v35",
    },
    {
        "id": "v36",
        "slug": "name-that-llm-sibling-style-v36",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v36_sibling_style",
        "version": "2",
        "message": "0.663 Pipeline + V40A Family-Only Stylometry Sibling Residual (5-Fold GroupKFold OOF Calibrated, Cross-Family Invariant) v36",
    },
    {
        "id": "v37",
        "slug": "name-that-llm-ntl-v41-grandmaster-horizon-v37",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v37_grandmaster_horizon",
        "version": "2",
        "message": "NTL v41 Grandmaster Horizon: Calibrated Pool Stylometry (0.50) + Expanded ATLAS 25% (Margin 0.35) + Soft Sinkhorn (0.50 / 0.40) v37",
    },
    {
        "id": "v38",
        "slug": "name-that-llm-horizon-sibling-v38",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v38_horizon_sibling",
        "version": "2",
        "message": "Grandmaster Horizon v41 + V40A Sibling Style Residual (max_alpha=0.06) + Zero-Violation Typography Guards v38",
    },
    {
        "id": "v39",
        "slug": "name-that-llm-consensus-blend-v39",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v39_consensus_blend",
        "version": "2",
        "message": "Grand Master Consensus: Balanced V31/V41 Multi-Modal Ensemble + Per-Row Member Normalization + Sibling Residual + Calibrated Sinkhorn (0.48) v39",
    },
]


def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m"] + cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def get_kernel_status(slug: str) -> str:
    res = run_cmd(["kaggle", "kernels", "status", f"dustinober/{slug}"])
    lines = [l for l in res.stdout.splitlines() if not l.startswith("Warning:")]
    out = "\n".join(lines).strip()
    out_lower = out.lower()
    if 'complete' in out_lower:
        return "COMPLETE"
    elif 'running' in out_lower:
        return "RUNNING"
    elif 'queued' in out_lower:
        return "QUEUED"
    elif 'error' in out_lower:
        return "ERROR"
    elif "404" in res.stderr or "notfound" in res.stderr.lower() or "404" in out or not out:
        return "NOT_PUSHED"
    return out or "UNKNOWN"


def get_submitted_messages() -> str:
    res = run_cmd(["kaggle", "competitions", "submissions", "-c", "name-that-llm"])
    return res.stdout


def submit_kernel(exp: dict) -> bool:
    console.print(f"[bold green]Submitting {exp['id']} (dustinober/{exp['slug']} v{exp['version']})...[/bold green]")
    res = run_cmd([
        "kaggle", "competitions", "submit",
        "-c", "name-that-llm",
        "-f", "submission.parquet",
        "-m", exp["message"],
        "-k", f"dustinober/{exp['slug']}",
        "-v", exp["version"]
    ])
    console.print(res.stdout.strip())
    if res.returncode != 0:
        console.print(f"[bold red]Submission failed for {exp['id']}:[/bold red] {res.stderr.strip()}")
        return False
    return True


def push_kernel(exp: dict) -> bool:
    console.print(f"[bold yellow]Pushing {exp['id']} ({exp['slug']})...[/bold yellow]")
    res = run_cmd(["kaggle", "kernels", "push", "-p", str(exp["dir"])])
    console.print(res.stdout.strip())
    if res.returncode != 0:
        console.print(f"[bold red]Push failed for {exp['id']}:[/bold red] {res.stderr.strip()}")
        return False
    m = re.search(r"Kernel version (\d+)", res.stdout)
    if m:
        exp["version"] = m.group(1)
    return True


def advance_pipeline(auto_push_and_submit: bool = False):
    console.rule("[bold cyan]5-Experiment SOTA Execution & Submission Status (v35 - v39)[/bold cyan]")

    sub_out = get_submitted_messages()
    lines = [l for l in sub_out.split("\n") if l.strip() and not l.startswith("Warning")][:10]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

    console.print("\n[bold yellow]5 Submissions Queue (v35 - v39):[/bold yellow]")
    active_gpu_commits = 0
    statuses = {}
    for exp in EXPERIMENT_QUEUE:
        st = get_kernel_status(exp["slug"])
        statuses[exp["id"]] = st
        if st in ("RUNNING", "QUEUED"):
            active_gpu_commits += 1
        already_sub = f" {exp['id']}" in sub_out
        sub_tag = "✅ SUBMITTED" if already_sub else "⏳ Pending Submit"
        console.print(f"  [{exp['id']}] {exp['slug']} (v{exp['version']}): [bold]{st}[/bold] | {sub_tag}")

    if not auto_push_and_submit:
        return

    # 1. Submit any COMPLETE kernels that haven't been submitted yet
    for exp in EXPERIMENT_QUEUE:
        already_sub = f" {exp['id']}" in sub_out
        if statuses[exp["id"]] == "COMPLETE" and not already_sub:
            if submit_kernel(exp):
                sub_out = get_submitted_messages()

    # 2. Push NOT_PUSHED kernels up to the 2 concurrent GPU commit limit
    for exp in EXPERIMENT_QUEUE:
        if active_gpu_commits >= 2:
            break
        if statuses[exp["id"]] == "NOT_PUSHED":
            if push_kernel(exp):
                active_gpu_commits += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--advance", action="store_true", help="Auto-submit completed kernels and push queued ones")
    args = parser.parse_args()
    advance_pipeline(auto_push_and_submit=args.advance)
