"""Automated Kaggle 5-Submission Pipeline Manager (v45 - v49).

Manages the 5-experiment SOTA queue:
- v45: SOTA 0.664 Peak Master (198k Open Pool + svc_w 16.0 + Sinkhorn 0.45 + Target Mix 0.35)
- v46: Open-Pool High-Precision ATLAS Synergy (198k Open Pool + svc_w 14.0 + Refined ATLAS Margin 0.28 / Weight 0.22)
- v47: Balanced Marginal Horizon (198k Open Pool + svc_w 12.0 + Sinkhorn 0.42 + Target Mix 0.32)
- v48: 198k Pool Sample Weight Optimization (_srcw 0.45 in SVC + svc_w 12.0 + Sinkhorn 0.45)
- v49: Grand Master Champion Consensus (198k Open Pool _srcw 0.40 + svc_w 15.0 + ATLAS Margin 0.28 / Weight 0.22 + Sinkhorn 0.45)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_BIN = shutil.which("kaggle") or "/Users/dustinober/.local/bin/kaggle"

EXPERIMENT_QUEUE = [
    {
        "id": "v45",
        "slug": "name-that-llm-open-pool-apex-16-v45",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v45_open_pool_apex_16",
        "version": "1",
        "message": "SOTA 0.664 Peak Master: 198k Open Pool + svc_w 16.0 + Sinkhorn 0.45 + Target Mix 0.35 v45",
    },
    {
        "id": "v46",
        "slug": "name-that-llm-atlas-precision-14-v46",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v46_atlas_precision_14",
        "version": "1",
        "message": "Open-Pool High-Precision ATLAS Synergy: 198k Open Pool + svc_w 14.0 + Refined ATLAS Margin 0.28 / Weight 0.22 v46",
    },
    {
        "id": "v47",
        "slug": "name-that-llm-sinkhorn-calibrated-v47",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v47_sinkhorn_calibrated",
        "version": "1",
        "message": "Balanced Marginal Horizon: 198k Open Pool + svc_w 12.0 + Sinkhorn 0.42 + Target Mix 0.32 v47",
    },
    {
        "id": "v48",
        "slug": "name-that-llm-pool-weight-tune-v48",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v48_pool_weight_tune",
        "version": "1",
        "message": "198k Pool Sample Weight Optimization: _srcw 0.45 in SVC + svc_w 12.0 + Sinkhorn 0.45 v48",
    },
    {
        "id": "v49",
        "slug": "name-that-llm-grandmaster-champion-v49",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v49_grandmaster_champion",
        "version": "1",
        "message": "Grand Master Champion Consensus: 198k Open Pool _srcw 0.40 + svc_w 15.0 + ATLAS Margin 0.28 / Weight 0.22 + Sinkhorn 0.45 v49",
    },
]


def run_kaggle_cmd(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [KAGGLE_BIN] + args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def get_kernel_status(slug: str) -> str:
    res = run_kaggle_cmd(["kernels", "status", f"dustinober/{slug}"])
    lines = [l for l in res.stdout.splitlines() if not l.startswith("Warning:")]
    out = "\n".join(lines).strip()
    out_lower = out.lower()
    if "complete" in out_lower:
        return "COMPLETE"
    elif "running" in out_lower:
        return "RUNNING"
    elif "queued" in out_lower:
        return "QUEUED"
    elif "error" in out_lower:
        return "ERROR"
    elif "denied" in res.stderr.lower() or "not found" in res.stderr.lower() or "404" in res.stderr or "denied" in out_lower or not out:
        return "NOT_PUSHED"
    return out or "UNKNOWN"


def get_submitted_messages() -> str:
    res = run_kaggle_cmd(["competitions", "submissions", "-c", "name-that-llm"])
    return res.stdout


def submit_kernel(exp: dict) -> bool:
    console.print(f"[bold green]Submitting {exp['id']} (dustinober/{exp['slug']} v{exp['version']})...[/bold green]")
    res = run_kaggle_cmd([
        "competitions", "submit",
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
    res = run_kaggle_cmd(["kernels", "push", "-p", str(exp["dir"])])
    console.print(res.stdout.strip())
    if res.returncode != 0:
        console.print(f"[bold red]Push failed for {exp['id']}:[/bold red] {res.stderr.strip()}")
        return False
    m = re.search(r"Kernel version (\d+)", res.stdout)
    if m:
        exp["version"] = m.group(1)
    else:
        exp["version"] = "1"
    return True


def is_already_submitted(exp: dict, sub_out: str) -> bool:
    for line in sub_out.splitlines():
        if exp["slug"] in line:
            return True
        if re.search(rf"\b{exp['id']}\s+SubmissionStatus", line):
            return True
    return False


def status_overview():
    sub_out = get_submitted_messages()
    lines = [l for l in sub_out.splitlines() if l.strip() and not l.startswith("Warning")][:10]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

    console.print("\n[bold yellow]5 Submissions Queue (v45 - v49):[/bold yellow]")
    for exp in EXPERIMENT_QUEUE:
        st = get_kernel_status(exp["slug"])
        already_sub = is_already_submitted(exp, sub_out)
        sub_tag = "✅ SUBMITTED" if already_sub else "⏳ Pending Submit"
        console.print(f"  [{exp['id']}] {exp['slug']} (v{exp['version']}): [bold]{st}[/bold] | {sub_tag}")


def run_pipeline_step():
    console.rule("[bold cyan]Executing Kaggle 5-Submission Pipeline (v45 - v49)[/bold cyan]")
    sub_out = get_submitted_messages()

    # Track statuses
    statuses = {}
    active_gpu_commits = 0
    for exp in EXPERIMENT_QUEUE:
        st = get_kernel_status(exp["slug"])
        statuses[exp["id"]] = st
        if st in ("RUNNING", "QUEUED"):
            active_gpu_commits += 1

    # 1. Submit any COMPLETE kernels not yet submitted
    for exp in EXPERIMENT_QUEUE:
        already_sub = is_already_submitted(exp, sub_out)
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
                statuses[exp["id"]] = "RUNNING"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--advance", action="store_true", help="Auto-submit completed kernels and push queued ones")
    args = parser.parse_args()

    if args.advance:
        run_pipeline_step()
    else:
        status_overview()
