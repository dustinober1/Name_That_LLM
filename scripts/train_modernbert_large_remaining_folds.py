"""Train remaining folds (1 to 4) of answerdotai/ModernBERT-large and aggregate full 5-fold OOF."""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

from name_that_llm.config import PROCESSED_DATA_DIR, TARGET_MODELS
from name_that_llm.data import load_train_data
from name_that_llm.metrics import compute_metrics, print_metrics_summary

console = Console()
MODEL_NAME = "answerdotai/ModernBERT-large"
MODEL_SLUG = MODEL_NAME.replace("/", "_")


def run_fold(fold_idx: int):
    console.rule(f"[bold cyan]Starting Training: Fold {fold_idx + 1}/5 ({MODEL_NAME})[/bold cyan]")
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_transformer.py",
        "--model-name",
        MODEL_NAME,
        "--fold-to-run",
        str(fold_idx),
        "--epochs",
        "3",
        "--batch-size",
        "2",
        "--accum-steps",
        "4",
        "--lr",
        "1.5e-5",
        "--max-length",
        "256",
    ]
    subprocess.run(cmd, check=True)
    console.print(f"[bold green]✓ Completed Fold {fold_idx + 1}/5![/bold green]")


def aggregate_oofs():
    console.rule("[bold cyan]Aggregating Full 5-Fold ModernBERT-large OOF Matrix[/bold cyan]")
    fold_dfs = []

    for f_idx in range(5):
        fold_file = PROCESSED_DATA_DIR / f"oof_{MODEL_SLUG}_fold{f_idx}.parquet"
        assert fold_file.exists(), f"Missing OOF file: {fold_file}"
        f_df = pd.read_parquet(fold_file)
        console.print(f"Loaded Fold {f_idx + 1} OOF ({len(f_df)} samples)")
        fold_dfs.append(f_df)

    combined_oof = pd.concat(fold_dfs, axis=0).sort_values("id").reset_index(drop=True)
    full_oof_path = PROCESSED_DATA_DIR / f"oof_{MODEL_SLUG}.parquet"
    combined_oof.to_parquet(full_oof_path, index=False)
    console.print(f"[bold green]✓ Saved full 5-fold OOF to {full_oof_path} ({len(combined_oof)} rows)[/bold green]")

    prob_cols = [f"prob_{m}" for m in TARGET_MODELS]
    P_base = combined_oof[prob_cols].values
    y_true = combined_oof["true_model"].values
    y_true_idx = np.array([TARGET_MODELS.index(m) for m in y_true])

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=combined_oof["pred_model"].values,
        y_prob=P_base,
        class_labels=TARGET_MODELS,
    )
    print_metrics_summary(metrics)

    # Master Ensemble Synergy Evaluation
    m_base_path = PROCESSED_DATA_DIR / "oof_answerdotai_ModernBERT-base.parquet"
    d_base_path = PROCESSED_DATA_DIR / "oof_microsoft_deberta-v3-base.parquet"
    hier_path = PROCESSED_DATA_DIR / "oof_hierarchical_specialist_v5.parquet"

    if m_base_path.exists() and d_base_path.exists() and hier_path.exists():
        console.rule("[bold magenta]Master Quad-Model Synergy Evaluation[/bold magenta]")
        m_base_df = pd.read_parquet(m_base_path).sort_values("id").reset_index(drop=True)
        d_base_df = pd.read_parquet(d_base_path).sort_values("id").reset_index(drop=True)
        hier_df = pd.read_parquet(hier_path).sort_values("id").reset_index(drop=True)

        P_ml = P_base
        P_mb = m_base_df[prob_cols].values
        P_db = d_base_df[prob_cols].values
        P_h  = hier_df[prob_cols].values

        # Quad-Model Blend: 25% ModernBERT-large + 25% ModernBERT-base + 30% DeBERTa Base + 20% Hierarchical
        P_quad = 0.25 * P_ml + 0.25 * P_mb + 0.30 * P_db + 0.20 * P_h
        raw_bal_acc = balanced_accuracy_score(y_true_idx, P_quad.argmax(axis=1))
        console.print(f"[bold yellow]Master Quad-Model Raw Bal Acc:[/bold yellow] {raw_bal_acc:.4f}")

        # Master Blend with Hungarian Prompt Matching
        console.print("[dim]Applying Hungarian prompt matching on master quad-model blend...[/dim]")
        hung_preds = np.zeros(len(combined_oof), dtype=int)
        for pid, grp in combined_oof.groupby("prompt_id"):
            idx = grp.index.values
            sub_P = P_quad[idx]
            c_log_p = np.log(np.clip(sub_P, 1e-12, 1.0))
            r, c = linear_sum_assignment(-c_log_p)
            hung_preds[idx[r]] = c

        hung_bal_acc = balanced_accuracy_score(y_true_idx, hung_preds)
        console.print(f"[bold green]★ Master Quad-Model + Hungarian Bal Acc:[/bold green] [bold cyan]{hung_bal_acc:.4f}[/bold cyan]")


def main():
    console.rule(f"[bold cyan]Full 5-Fold Training Pipeline: {MODEL_NAME}[/bold cyan]")

    # Check which folds are already completed
    for fold_idx in range(1, 5):
        fold_file = PROCESSED_DATA_DIR / f"oof_{MODEL_SLUG}_fold{fold_idx}.parquet"
        ckpt_dir = Path("models/checkpoints") / f"{MODEL_SLUG}_fold{fold_idx}"

        if fold_file.exists() and ckpt_dir.exists():
            console.print(f"[bold yellow]Fold {fold_idx + 1}/5 already completed. Skipping.[/bold yellow]")
            continue

        run_fold(fold_idx)

    aggregate_oofs()


if __name__ == "__main__":
    main()
