"""Train remaining folds (1 to 4) of microsoft/deberta-v3-base and aggregate full 5-fold OOF."""

import subprocess
import sys
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

from name_that_llm.config import PROCESSED_DATA_DIR, TARGET_MODELS
from name_that_llm.data import load_train_data
from name_that_llm.metrics import compute_metrics, print_metrics_summary

console = Console()
MODEL_NAME = "microsoft/deberta-v3-base"
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
        "4",
        "--accum-steps",
        "2",
        "--lr",
        "2e-5",
        "--head-lr",
        "1e-4",
        "--max-length",
        "256",
    ]
    res = subprocess.run(cmd, check=True)
    console.print(f"[bold green]✓ Completed Fold {fold_idx + 1}/5![/bold green]")


def aggregate_oofs():
    console.rule("[bold cyan]Aggregating Full 5-Fold DeBERTa-v3 Base OOF Matrix[/bold cyan]")
    df_train = load_train_data()
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

    # Evaluate Synergy with Hierarchical Specialist
    hier_path = PROCESSED_DATA_DIR / "oof_hierarchical_specialist_v5.parquet"
    if hier_path.exists():
        console.rule("[bold magenta]Master Multi-Model Synergy Evaluation[/bold magenta]")
        hier_df = pd.read_parquet(hier_path).sort_values("id").reset_index(drop=True)
        P_hier = hier_df[prob_cols].values

        # 50/50 Raw Master Blend
        P_blend = 0.50 * P_base + 0.50 * P_hier
        raw_bal_acc = balanced_accuracy_score(y_true_idx, P_blend.argmax(axis=1))
        console.print(f"[bold yellow]50/50 Raw Master Blend Bal Acc:[/bold yellow] {raw_bal_acc:.4f}")

        # Master Blend with Hungarian Prompt Matching
        console.print("[dim]Applying Hungarian prompt matching on master blend...[/dim]")
        hung_preds = np.zeros(len(combined_oof), dtype=int)
        for pid, grp in combined_oof.groupby("prompt_id"):
            idx = grp.index.values
            sub_P = P_blend[idx]
            c_log_p = np.log(np.clip(sub_P, 1e-12, 1.0))
            r, c = linear_sum_assignment(-c_log_p)
            hung_preds[idx[r]] = c

        hung_bal_acc = balanced_accuracy_score(y_true_idx, hung_preds)
        console.print(f"[bold green]★ Master Blend + Hungarian Bal Acc: {hung_bal_acc:.4f}[/bold green]")


def main():
    for f in [1, 2, 3, 4]:
        run_fold(f)
    aggregate_oofs()


if __name__ == "__main__":
    main()
