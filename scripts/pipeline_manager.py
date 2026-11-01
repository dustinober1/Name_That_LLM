"""Automated Kaggle 5-Submission Pipeline Manager (v40 - v44).

Manages the 5-experiment SOTA queue:
- v40: Verified SOTA 0.664 Reproduction (Softened Typography Evidence 1.2x + Marginally Balanced Sinkhorn 0.60)
- v41: 198k Clean Open Pool SOTA (crtproduction/name-that-llm-open-pool + svc_w 12.0)
- v42: Grandmaster Synergy (198k Open Pool + Softened Typography 1.2x + Calibrated Sinkhorn 0.55 + svc_w 4.0)
- v43: Open-Pool Calibrated ATLAS (198k Pool + Refined ATLAS Margin 0.28 / Weight 0.22 + Sinkhorn 0.58)
- v44: Horizon Master Consensus (198k Pool + Typography 1.2x + Family Invariant Sibling Residual + Calibrated Sinkhorn 0.50)
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
        "id": "v40",
        "slug": "name-that-llm-sota-0664-tuned-v40",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v40_sota_0664_tuned",
        "version": "1",
        "message": "Verified SOTA 0.664 Reproduction: Softened Typography Evidence (1.2x) + Marginally Balanced Sinkhorn (0.60) v40",
    },
    {
        "id": "v41",
        "slug": "name-that-llm-open-pool-198k-v41",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v41_open_pool_198k",
        "version": "1",
        "message": "198k Clean Open Pool SOTA: crtproduction/name-that-llm-open-pool (198k Clean Passages) + svc_w 12.0 v41",
    },
    {
        "id": "v42",
        "slug": "name-that-llm-open-pool-synergy-v42",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v42_open_pool_tuned_synergy",
        "version": "1",
        "message": "Grandmaster Synergy: 198k Open Pool + Softened Typography (1.2x) + Calibrated Sinkhorn (0.55) + svc_w 4.0 v42",
    },
    {
        "id": "v43",
        "slug": "name-that-llm-atlas-calibrated-v43",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v43_atlas_calibrated",
        "version": "1",
        "message": "Open-Pool Calibrated ATLAS: 198k Pool + Refined ATLAS Margin (0.28) / Weight (0.22) + Sinkhorn (0.58) v43",
    },
    {
        "id": "v44",
        "slug": "name-that-llm-master-consensus-v44",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v44_master_consensus",
        "version": "1",
        "message": "Horizon Master Consensus: 198k Pool + Typography (1.2x) + Family Invariant Sibling Residual + Calibrated Sinkhorn (0.50) v44",
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
    sub_out = get_submitted_messages()
    lines = [l for l in sub_out.splitlines() if l.strip() and not l.startswith("Warning")][:10]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

def status_overview():
    sub_out = get_submitted_messages()
    lines = [l for l in sub_out.splitlines() if l.strip() and not l.startswith("Warning")][:10]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

    console.print("\n[bold yellow]5 Submissions Queue (v40 - v44):[/bold yellow]")
    for exp in EXPERIMENT_QUEUE:
        st = get_kernel_status(exp["slug"])
        already_sub = is_already_submitted(exp, sub_out)
        sub_tag = "✅ SUBMITTED" if already_sub else "⏳ Pending Submit"
        console.print(f"  [{exp['id']}] {exp['slug']} (v{exp['version']}): [bold]{st}[/bold] | {sub_tag}")


def run_pipeline_step():
    console.rule("[bold cyan]Executing Kaggle 5-Submission Pipeline (v40 - v44)[/bold cyan]")
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
