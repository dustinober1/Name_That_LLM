"""Automated Kaggle 5-Submission Pipeline Manager (v30 - v34).

Manages the 5-experiment SOTA queue:
- v30 (ver 3): Verified 0.654 Pipeline (Full nlm-v27 + Llama-3.1-8B v5 + 15% Comp Stack + 10% ATLAS + v38a 34-Marker Bayesian Typography)
- v31 (ver 2): 0.654 Pipeline + v39 Per-Row Active Ensemble Member Normalization (0.5 global + 0.5 row-active before Sinkhorn)
- v32 (ver 1): Upgraded 120k-Pool Qwen2.5-14B v5 6-Fold Ensemble (dhruvtibarewal/nlm-q25-v5, w=2.0) + 0.654 Typography + Per-Row Norm
- v33 (ver 1): v32 + 3rd Shared-Prompt Gaussian Head (q25_14v5 gauss=0.45) + GLM/Kimi 1,516-Passage Pool Patch + Fixed Soft SVC Topic Bias (LP['svc4'], alpha=0.62)
- v34 (ver 1): Grand Hybrid (Dual Qwen2.5-14B: q25_14v5 w=2.0 + q25_14 w=0.5) + Pool-Augmented Bayesian Typography Prior (75% Train + 25% 120k Pool) + Zero-Violation Typography Guards + Sinkhorn 0.85
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
        "id": "v30",
        "slug": "name-that-llm-nlm-v27-glm-kimi-v30",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v27_glm_kimi_v30",
        "version": "3",
        "message": "SOTA 0.654 Verified Reproduction (Full nlm-v27 + Llama-3.1-8B v5 + 15% Comp Stack + 10% ATLAS + 34-Marker Bayesian Typography) v30",
    },
    {
        "id": "v31",
        "slug": "name-that-llm-nlm-v31-topic-bias",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v31_topic_bias",
        "version": "2",
        "message": "SOTA 0.654 + Per-Row Active Ensemble Member Normalization (0.5 Global + 0.5 Row-Active) + Bayesian Typography v31",
    },
    {
        "id": "v32",
        "slug": "name-that-llm-nlm-v32-spectral-consensus",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v32_spectral_consensus",
        "version": "1",
        "message": "Upgraded 120k-Pool Qwen2.5-14B v5 6-Fold Ensemble (nlm-q25-v5, w=2.0) + Full nlm-v27 + 0.654 Typography + Per-Row Norm v32",
    },
    {
        "id": "v33",
        "slug": "name-that-llm-nlm-v33-sharper-blend",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v33_sharper_blend",
        "version": "1",
        "message": "Tri-Gaussian (q14v5 + q8v5 + q25_14v5 gauss=0.45) + GLM/Kimi Pool Patch + Fixed Soft SVC Topic Bias (svc4, alpha=0.62) + 0.654 Typography v33",
    },
    {
        "id": "v34",
        "slug": "name-that-llm-nlm-v34-grand-hybrid",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v34_grand_hybrid",
        "version": "1",
        "message": "Grand Hybrid: Dual Qwen2.5-14B (q25_14v5 w=2.0 + q25_14 w=0.5) + Pool-Augmented Bayesian Typography Prior + Zero-Violation Guards + Sinkhorn 0.85 v34",
    },
]


def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def get_kernel_status(slug: str, username: str = "dustinober") -> str:
    res = run_cmd(["kaggle", "kernels", "status", f"{username}/{slug}"])
    out = (res.stdout + res.stderr).lower()
    if "kernelworkerstatus.complete" in out:
        return "COMPLETE"
    elif "kernelworkerstatus.running" in out:
        return "RUNNING"
    elif "kernelworkerstatus.queued" in out:
        return "QUEUED"
    elif "kernelworkerstatus.failed" in out or "kernelworkerstatus.error" in out:
        return "FAILED"
    elif "not found" in out or "404" in out or "cannot access kernel" in out:
        return "NOT_PUSHED"
    return "UNKNOWN"


def get_submitted_messages() -> str:
    sub_res = run_cmd(["kaggle", "competitions", "submissions", "name-that-llm"])
    return sub_res.stdout


def submit_kernel(exp: dict, username: str = "dustinober") -> bool:
    kernel_id = f"{username}/{exp['slug']}"
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "name-that-llm",
        "-f",
        "submission.parquet",
        "-k",
        kernel_id,
        "-v",
        exp["version"],
        "-m",
        exp["message"],
    ]
    console.print(f"[bold cyan]Submitting {exp['id']} ({kernel_id} v{exp['version']})...[/bold cyan]")
    res = run_cmd(cmd)
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
    console.rule("[bold cyan]5-Experiment SOTA Execution & Submission Status (v30 - v34)[/bold cyan]")

    sub_out = get_submitted_messages()
    lines = [l for l in sub_out.split("\n") if l.strip() and not l.startswith("Warning")][:10]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

    console.print("\n[bold yellow]5 Submissions Queue (v30 - v34):[/bold yellow]")
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
