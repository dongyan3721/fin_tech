"""训练脚本：支持参数化训练 + Agent 开关 + 时序模型替换。

时序编码器选项（通过 --temporal 指定）:
    gated_conv   - 门控时间卷积（默认，Legacy 对齐）
    gru          - GRU 循环神经网络
    lstm         - LSTM 长短期记忆网络
    transformer  - Transformer 自注意力机制

用法示例:
    # ==================== 基础训练 ====================
    # 默认参数训练（无 Agent，300 epochs，gated_conv）
    .venv\\Scripts\\python.exe scripts/train.py

    # 指定 epochs 和学习率
    .venv\\Scripts\\python.exe scripts/train.py --epochs 500 --lr 0.0005

    # ==================== Agent 控制 ====================
    # 启用反思 Agent（增强点 D：难例重加权）
    .venv\\Scripts\\python.exe scripts/train.py --agent

    # 启用 Agent 但仅诊断，不做第二轮重训
    .venv\\Scripts\\python.exe scripts/train.py --agent --no-reflection-retrain

    # ==================== 时序模型消融 ====================
    # 使用 GRU 作为时序编码器
    .venv\\Scripts\\python.exe scripts/train.py --temporal gru --epochs 300

    # 使用 LSTM 作为时序编码器
    .venv\\Scripts\\python.exe scripts/train.py --temporal lstm --epochs 300

    # 使用 Transformer 作为时序编码器
    .venv\\Scripts\\python.exe scripts/train.py --temporal transformer --epochs 300

    # ==================== 输出控制 ====================
    # 自定义输出目录名（便于对比实验）
    .venv\\Scripts\\python.exe scripts/train.py --name exp_gru_baseline
    .venv\\Scripts\\python.exe scripts/train.py --name exp_lstm_agent --agent --temporal lstm

    # 指定日志文件
    .venv\\Scripts\\python.exe scripts/train.py --epochs 300 --agent --log logs/train_001.log

    # ==================== 完整参数示例 ====================
    .venv\\Scripts\\python.exe scripts/train.py \\
        --epochs 500 \\
        --lr 0.001 \\
        --weight-decay 1e-5 \\
        --hidden-dim 128 \\
        --dropout 0.3 \\
        --temporal gru \\
        --agent \\
        --name exp_gru_d128_agent \\
        --log logs/exp_gru_d128_agent.log
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 统一 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.current.config import CONFIG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TGC 供应链风控模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--epochs", type=int, default=500, help="训练轮数 (default: 500)")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率 (default: 0.001)")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="L2 正则化 (default: 1e-5)")
    p.add_argument("--hidden-dim", type=int, default=64, help="隐藏层维度 (default: 64)")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout 比率 (default: 0.3)")
    p.add_argument("--temporal", type=str, default="gated_conv",
                   choices=["gated_conv", "gru", "lstm", "transformer"],
                   help="时序编码器 (default: gated_conv)")

    p.add_argument("--label-scheme", type=str, default=None,
                   choices=["kmv", "hybrid"],
                   help="标签方案：kmv=基线简化KMV, hybrid=方案D混合标签 (default: 取 config)")
    p.add_argument("--no-prepare-label", action="store_true",
                   help="训练前不重新生成标签，直接使用磁盘已有的 processed 标签")

    p.add_argument("--agent", action="store_true", help="启用反思 Agent（增强点 D）")
    p.add_argument("--no-agent", action="store_true", help="禁用 Agent（默认）")
    p.add_argument("--no-reflection-retrain", action="store_true",
                   help="启用 Agent 分析但不做第二轮加权重训（仅输出诊断报告）")

    p.add_argument("--name", type=str, default=None,
                   help="自定义输出目录名（如 exp_gru_baseline）")
    p.add_argument("--eval-every", type=int, default=100,
                   help="每 N 轮评估一次并保存最佳模型 (0=不评估, default: 100)")
    p.add_argument("--no-resume", action="store_true", help="采集时不使用断点续跑")
    p.add_argument("--log", type=str, default=None, help="日志输出文件路径")

    return p.parse_args()


def build_config(args: argparse.Namespace):
    """根据命令行参数构建配置。"""
    cfg = deepcopy(CONFIG)

    cfg.model.epochs = args.epochs
    cfg.model.lr = args.lr
    cfg.model.weight_decay = args.weight_decay
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.dropout = args.dropout
    cfg.model.temporal_encoder = args.temporal

    if args.label_scheme is not None:
        cfg.labels.label_scheme = args.label_scheme

    if args.agent and not args.no_agent:
        cfg.agent.enabled = True
        cfg.agent.hook = "reflection"
        if args.no_reflection_retrain:
            cfg.agent.reflection.enabled = False
    else:
        cfg.agent.enabled = False

    return cfg


def print_config(cfg, args: argparse.Namespace) -> None:
    """打印训练配置摘要。"""
    print("=" * 60)
    print("TGC 供应链风控模型训练")
    print("=" * 60)
    print(f"  时间:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  训练轮数:    {cfg.model.epochs}")
    print(f"  学习率:      {cfg.model.lr}")
    print(f"  权重衰减:    {cfg.model.weight_decay}")
    print(f"  隐藏层维度:  {cfg.model.hidden_dim}")
    print(f"  Dropout:     {cfg.model.dropout}")
    print(f"  时序编码器:  {cfg.model.temporal_encoder}")
    print(f"  标签方案:    {cfg.labels.label_scheme}")
    print(f"  标签变换:    {cfg.model.label_transform}")
    print(f"  建图方案:    {cfg.model.graph_scheme} (lag={cfg.model.graph_lag})")
    print(f"  Agent:       {'反思 Agent' if cfg.agent.enabled else '关闭'}")
    print(f"  评估间隔:    {args.eval_every} 轮" if args.eval_every > 0 else "  评估间隔:    关闭")
    print("=" * 60)
    print()


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """打印评估指标表格。"""
    if not metrics:
        return
    print(f"\n{prefix}评估指标:")
    print("-" * 40)
    print(f"  MSE:         {metrics.get('mse', 0):.6f}")
    print(f"  MAE:         {metrics.get('mae', 0):.6f}")
    print(f"  RMSE:        {metrics.get('rmse', 0):.6f}")
    print(f"  R²(prob):    {metrics.get('r2', 0):.4f}")
    print(f"  R²(logit):   {metrics.get('r2_logit', 0):.4f}")
    print(f"  Spearman:    {metrics.get('spearman', 0):.4f}")
    print(f"  IC(Pearson): {metrics.get('ic', 0):.4f}")
    if metrics.get('auc') is not None:
        print(f"  AUC:         {metrics.get('auc', 0):.4f}")
        print(f"  KS:          {metrics.get('ks', 0):.4f}  (违约样本 {metrics.get('n_default', 0)})")
    print(f"  测试样本数:  {metrics.get('n_test', 0)}")
    if 'rating_accuracy' in metrics:
        print(f"  评级准确率:  {metrics.get('rating_accuracy', 0):.3f} "
              f"({metrics.get('rating_correct', 0)}/{metrics.get('rating_total', 0)})")
    print("-" * 40)


def redirect_logs(log_path: str) -> None:
    """重定向日志到文件。"""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = f
    sys.stderr = f


def prepare_labels(cfg) -> None:
    """训练前按 cfg.labels.label_scheme 重新生成标签并导出到 processed。

    保证 processed/labels.parquet 与本次训练选择的标签方案一致（避免磁盘残留其他方案）。
    """
    import src.current.labels  # noqa: F401  触发 labeler/scheme 注册
    from src.current.labels.base import generate_labels
    from src.current.transform.exporter import export_labels

    print(f"[label] 训练前按方案 {cfg.labels.label_scheme!r} 重新生成标签 ...")
    generate_labels(cfg.labels.label_scheme)
    export_labels()


def main() -> int:
    args = parse_args()

    if args.log:
        redirect_logs(args.log)

    cfg = build_config(args)
    print_config(cfg, args)

    if not args.no_prepare_label:
        prepare_labels(cfg)

    from src.current.train.trainer import Trainer

    trainer = Trainer(config=cfg, run_name=args.name, eval_every=args.eval_every)
    result = trainer.run()

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  输出目录: {result.get('out_dir')}")

    metrics = {k: v for k, v in result.items() if k != "out_dir"}
    print_metrics(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
