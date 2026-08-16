"""训练轮数搜索实验：每 100 轮评估一次，找到最优训练轮数。

用法:
    .venv\Scripts\python.exe scripts/search_epochs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.current.train.trainer import Trainer
from src.current.config import CONFIG
from copy import deepcopy


def main():
    max_epochs = 2000
    eval_every = 100

    cfg = deepcopy(CONFIG)
    cfg.model.epochs = max_epochs
    cfg.agent.enabled = False

    print("=" * 60)
    print(f"训练轮数搜索实验: {max_epochs} 轮, 每 {eval_every} 轮评估")
    print("=" * 60)

    trainer = Trainer(config=cfg, run_name="search_epochs_2000", eval_every=eval_every)
    result = trainer.run()

    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)

    eval_log_path = trainer.out_dir / "eval_log.csv"
    if eval_log_path.exists():
        import pandas as pd
        df = pd.read_csv(eval_log_path)
        print(f"\n共 {len(df)} 个评估点:")
        print(df.to_string(index=False))

        # 找最佳 R²
        best_r2_idx = df['r2'].idxmax()
        best_r2_row = df.loc[best_r2_idx]
        print(f"\n最佳 R²(prob) 在 epoch {int(best_r2_row['epoch'])}:")
        print(f"  R²(prob) = {best_r2_row['r2']:.4f}")
        print(f"  MSE = {best_r2_row['mse']:.6f}")
        print(f"  MAE = {best_r2_row['mae']:.6f}")
        print(f"  Spearman = {best_r2_row['spearman']:.4f}")
        print(f"  IC = {best_r2_row['ic']:.4f}")

        # 找最佳 IC
        best_ic_idx = df['ic'].idxmax()
        best_ic_row = df.loc[best_ic_idx]
        print(f"\n最佳 IC(Pearson) 在 epoch {int(best_ic_row['epoch'])}:")
        print(f"  R²(prob) = {best_ic_row['r2']:.4f}")
        print(f"  IC = {best_ic_row['ic']:.4f}")

        # 找最佳 Spearman
        best_sp_idx = df['spearman'].idxmax()
        best_sp_row = df.loc[best_sp_idx]
        print(f"\n最佳 Spearman 在 epoch {int(best_sp_row['epoch'])}:")
        print(f"  R²(prob) = {best_sp_row['r2']:.4f}")
        print(f"  Spearman = {best_sp_row['spearman']:.4f}")

        print(f"\n建议默认训练轮数 (按 R²): {int(best_r2_row['epoch'])}")


if __name__ == "__main__":
    main()
