"""Automated Kaggle Multi-Experiment Submission & GPU Pipeline Manager.

Manages the 5-experiment queue:
- v29: SOTA 0.651 nlm-v27-nollama (running)
- v30: SOTA 0.651 nlm-v27 + GLM/Kimi Rare-Class Pool Patch (running)
- v31: Enhanced 0.651 with Soft Topic Bias Attenuation (queued)
- v32: Dual-Stream LoRA v5 + 8-Head Augmented Spectral Bank (queued)
- v33: Sharper Blend & Margin-Optimized ATLAS Allocation (queued)
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_QUEUE = [
    {
        "id": "v29",
        "slug": "name-that-llm-nlm-v27-nollama-v29",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v27_nollama",
        "message": "SOTA 0.651 nlm-v27-nollama (Dhruv v5 LoRAs + Gaussian Heads + ATLAS) v29",
    },
    {
        "id": "v30",
        "slug": "name-that-llm-nlm-v27-glm-kimi-v30",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v27_glm_kimi_v30",
        "message": "SOTA nlm-v27 + GLM/Kimi Rare-Class Pool Patch v30",
    },
    {
        "id": "v31",
        "slug": "name-that-llm-nlm-v31-topic-bias",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v31_topic_bias",
        "message": "Enhanced 0.651 nlm-v27 + Soft Topic Bias Attenuation v31",
    },
    {
        "id": "v32",
        "slug": "name-that-llm-nlm-v32-spectral-consensus",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v32_spectral_consensus",
        "message": "Dual-Stream LoRA v5 + 8-Head Augmented Spectral Style Bank v32",
    },
    {
        "id": "v33",
        "slug": "name-that-llm-nlm-v33-sharper-blend",
        "dir": PROJECT_ROOT / "kernels" / "nlm_v33_sharper_blend",
        "message": "Sharper Temperature Blend & Margin-Adaptive ATLAS Allocation v33",
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
    elif "not found" in out or "404" in out:
        return "NOT_PUSHED"
    return "UNKNOWN"


def check_pipeline_status():
    console.rule("[bold cyan]5-Experiment Execution & Submission Status[/bold cyan]")
    
    # 1. Check competition submissions
    sub_res = run_cmd(["kaggle", "competitions", "submissions", "name-that-llm"])
    lines = [l for l in sub_res.stdout.split("\n") if l.strip() and not l.startswith("Warning")][:7]
    console.print("[bold green]Recent Competition Submissions:[/bold green]")
    for l in lines:
        console.print(f"  {l}")

    console.print("\n[bold yellow]5 Next Experiments Queue:[/bold yellow]")
    for exp in EXPERIMENT_QUEUE:
        st = get_kernel_status(exp["slug"])
        console.print(f"  [{exp['id']}] {exp['slug']}: [bold]{st}[/bold]")


if __name__ == "__main__":
    check_pipeline_status()
