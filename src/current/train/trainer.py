"""TGC 训练/评估/预测编排（端到端复现 legacy 效果）。

产物落 repository/outputs/current_tgc_<时间戳>/：
  train_dataset.csv / test_dataset.csv / test_predictions.csv / 2025_predictions.csv
  training_loss.png / prediction_scatter.png / graph_*.csv
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.current.config import CONFIG
from src.current.models.tgc import TGCModel
from src.current.train.dataset import (DatasetBundle, build_dataset,
                                       build_graph, build_graph_by_pred_year)
from src.current.viz.base import VizManager

_EPS = 1e-4  # logit 变换裁剪，避免 log(0)


def _logit(x: np.ndarray) -> np.ndarray:
    """概率 -> logit(实数)，用于在对数几率空间评估 R²（不受开关影响）。"""
    xc = np.clip(np.asarray(x, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(xc / (1 - xc))


def _prob_to_rating(prob: float) -> str:
    """将违约概率映射为信用评级 (AAA~D)。"""
    thresholds = [(0.01, "AAA"), (0.05, "AA"), (0.1, "A"), (0.2, "BBB"),
                  (0.3, "BB"), (0.4, "B"), (0.6, "CCC")]
    for t, r in thresholds:
        if prob < t:
            return r
    return "D"


def _rating_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """评估评级分类准确性：比较实际评级 vs 预测评级。"""
    actual_ratings = [_prob_to_rating(p) for p in actual]
    pred_ratings = [_prob_to_rating(p) for p in predicted]

    total = len(actual_ratings)
    correct = sum(1 for a, p in zip(actual_ratings, pred_ratings) if a == p)
    accuracy = correct / total if total > 0 else 0

    # 按评级分组统计
    rating_classes = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    per_class = {}
    for cls in rating_classes:
        cls_indices = [i for i, r in enumerate(actual_ratings) if r == cls]
        if not cls_indices:
            continue
        cls_correct = sum(1 for i in cls_indices if pred_ratings[i] == cls)
        per_class[cls] = {
            "count": len(cls_indices),
            "correct": cls_correct,
            "accuracy": cls_correct / len(cls_indices),
        }

    # 混淆矩阵（简化版：只统计预测正确的分布）
    confusion = {}
    for actual_cls in rating_classes:
        for pred_cls in rating_classes:
            count = sum(1 for a, p in zip(actual_ratings, pred_ratings)
                       if a == actual_cls and p == pred_cls)
            if count > 0:
                confusion[f"{actual_cls}->{pred_cls}"] = count

    return {
        "rating_accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class": per_class,
        "confusion": confusion,
    }


def _safe_corr(func, a, b) -> float:
    """计算相关系数，样本过少/常数序列时返回 nan 安全值。"""
    try:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        r = func(a, b)[0]
        return float(r)
    except Exception:
        return float("nan")


class Trainer:
    def __init__(self, config=CONFIG, run_name: Optional[str] = None,
                 eval_every: int = 0) -> None:
        self.cfg = config
        self.scaler = StandardScaler()
        self.model: Optional[TGCModel] = None
        self.viz = VizManager()
        if run_name:
            self.out_dir: Path = self.cfg.outputs_dir / run_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir: Path = self.cfg.outputs_dir / f"current_tgc_{ts}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._agent = self._maybe_agent()
        # logit 标签变换开关（缓解低方差）
        self.logit = (self.cfg.model.label_transform == "logit")
        self._final_activation = "identity" if self.logit else "sigmoid"
        self.eval_every = eval_every  # 定期评估间隔（0=不评估）

    # -- 标签变换 ---------------------------------------------------------
    def _to_train_target(self, y: np.ndarray) -> np.ndarray:
        if self.logit:
            yc = np.clip(y, _EPS, 1 - _EPS)
            return np.log(yc / (1 - yc))
        return y

    def _to_prob(self, pred: np.ndarray) -> np.ndarray:
        if self.logit:
            return 1.0 / (1.0 + np.exp(-pred))
        return pred

    def _build_graph(self, edges, companies, pred_years):
        """按 config 选择建图方案。"""
        if self.cfg.model.graph_scheme == "densest_legacy":
            return build_graph(edges, companies)  # 已废弃，仅对照
        return build_graph_by_pred_year(edges, companies, pred_years,
                                        lag=self.cfg.model.graph_lag)

    def _maybe_agent(self):
        if not self.cfg.agent.enabled:
            return None
        from src.current.registry import AGENT_HOOKS
        import src.current.agents  # noqa: F401  确保注册
        try:
            return AGENT_HOOKS.create(self.cfg.agent.hook)
        except KeyError as e:
            print(f"[agent] 未找到 hook {self.cfg.agent.hook}: {e}")
            return None

    # -- 核心流程 ---------------------------------------------------------
    def run(self) -> dict:
        bundle = build_dataset()
        if len(bundle.train.X) == 0:
            raise RuntimeError("训练集为空，无法训练。请检查 nodes/labels 是否已生成。")

        self._save_split_csv(bundle)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[train] 设备: {self.device}" + (f" ({torch.cuda.device_count()}x {torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))

        n, t, f = bundle.train.X.shape
        Xtr = self.scaler.fit_transform(bundle.train.X.reshape(-1, f)).reshape(n, t, f)
        x_tr = torch.tensor(Xtr, dtype=torch.float, device=self.device)
        y_tr = torch.tensor(self._to_train_target(bundle.train.y), dtype=torch.float, device=self.device).view(-1, 1)

        tr_ei, tr_ew = self._build_graph(bundle.edges, bundle.train.companies, bundle.train.pred_years)
        if tr_ei is not None:
            tr_ei, tr_ew = tr_ei.to(self.device), tr_ew.to(self.device)
        use_gcn = tr_ei is not None
        print(f"[train] 建图方案={self.cfg.model.graph_scheme} | 标签变换={self.cfg.model.label_transform} | "
              f"模式: {'GCNConv(有图)' if use_gcn else '简化卷积(无图)'}")
        self.viz.dispatch("graph", {"nodes": bundle.nodes, "edges": bundle.edges}, self.out_dir)

        # ========== 第一轮：均匀权重训练 ==========
        print("[train] === 第一轮训练（均匀权重）===")
        self.model = TGCModel(input_dim=f, cfg=self.cfg.model, use_gcn=use_gcn,
                              final_activation=self._final_activation).to(self.device)
        train_result = self._train_loop(x_tr, y_tr, tr_ei, tr_ew, bundle=bundle, eval_every=self.eval_every)
        losses_r1 = train_result['losses']
        self.viz.dispatch("training", {"losses": losses_r1}, self.out_dir)

        # 如果有最佳 checkpoint，加载它
        if train_result.get('best_state') is not None:
            self.model.load_state_dict(train_result['best_state'])
            print(f"[train] 加载最佳 epoch {train_result['best_epoch']} 的模型 (R²={train_result['best_metric']:.4f})")

        metrics_r1 = {}
        test_preds_r1 = None
        if len(bundle.test.X) > 0:
            metrics_r1, test_preds_r1 = self._evaluate(bundle)

        # ========== 第二轮：反思 Agent 加权重训 ==========
        reflection_result = None
        has_reflection = (
            self._agent is not None
            and self.cfg.agent.reflection.enabled
            and test_preds_r1 is not None
            and hasattr(self._agent, "reflect")
        )

        if has_reflection:
            print("\n[train] === 反思 Agent 分析 ===")
            reflection_result = self._agent.reflect(
                test_preds=test_preds_r1,
                nodes=bundle.nodes,
                edges=bundle.edges,
                feature_columns=bundle.feature_columns,
                bundle=bundle,
            )
            sample_weights = reflection_result["weights"]

            reflection_report = reflection_result.get("report", {})
            report_path = self.out_dir / "reflection_report.json"
            report_path.write_text(
                json.dumps(reflection_report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"[reflection] 报告已保存: {report_path}")

            print("\n[train] === 第二轮训练（反思加权）===")
            self.model = TGCModel(input_dim=f, cfg=self.cfg.model, use_gcn=use_gcn,
                                  final_activation=self._final_activation).to(self.device)
            train_result_r2 = self._train_loop(x_tr, y_tr, tr_ei, tr_ew,
                                               sample_weight=sample_weights,
                                               bundle=bundle, eval_every=self.eval_every)
            losses_r2 = train_result_r2['losses']

            if train_result_r2.get('best_state') is not None:
                self.model.load_state_dict(train_result_r2['best_state'])
                print(f"[train] 加载最佳 epoch {train_result_r2['best_epoch']} 的模型")

            metrics = {}
            if len(bundle.test.X) > 0:
                metrics, test_preds = self._evaluate(bundle)
        else:
            metrics = metrics_r1
            losses_r2 = losses_r1

        # 保存评估日志
        if train_result.get('eval_log'):
            eval_log_df = pd.DataFrame(train_result['eval_log'])
            eval_log_df.to_csv(self.out_dir / "eval_log.csv", index=False, encoding="utf-8-sig")

        if bundle.future is not None:
            self._predict_future(bundle)

        run_info = {
            "n_train": int(len(bundle.train.X)),
            "n_test": int(len(bundle.test.X)),
            "n_future": int(len(bundle.future.X)) if bundle.future is not None else 0,
            "use_gcn": bool(use_gcn),
            "train_sample_edges": int(tr_ei.shape[1]) if tr_ei is not None else 0,
            "reflection_enabled": has_reflection,
        }
        # 处理 train_result 返回值（dict 或 list）
        if isinstance(train_result, dict):
            run_info["final_train_loss"] = float(train_result['losses'][-1]) if train_result.get('losses') else None
            if train_result.get('best_epoch'):
                run_info["best_epoch"] = train_result['best_epoch']
                run_info["best_r2"] = float(train_result['best_metric'])
        else:
            run_info["final_train_loss"] = float(train_result[-1]) if train_result else None

        if has_reflection and metrics_r1:
            run_info["r1_spearman"] = metrics_r1.get("spearman")
            run_info["r1_mae"] = metrics_r1.get("mae")
            run_info["r1_r2"] = metrics_r1.get("r2")
            if reflection_result and "metrics" in reflection_result:
                run_info["reflection_mae"] = reflection_result["metrics"].get("mean_abs_error")

        self._save_metrics(metrics, run_info)
        self._save_checkpoint(bundle, use_gcn, f)

        print(f"[train] 全部产物已保存到: {self.out_dir}")
        return {"out_dir": str(self.out_dir), **metrics}

    def _save_checkpoint(self, bundle: DatasetBundle, use_gcn: bool, input_dim: int) -> None:
        """保存模型 checkpoint，供推理脚本加载。"""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "input_dim": input_dim,
            "use_gcn": use_gcn,
            "final_activation": self._final_activation,
            "feature_columns": bundle.feature_columns,
            "model_config": {
                "temporal_encoder": self.cfg.model.temporal_encoder,
                "hidden_dim": self.cfg.model.hidden_dim,
                "dropout": self.cfg.model.dropout,
                "temporal_kernel": self.cfg.model.temporal_kernel,
            },
        }
        ckpt_path = self.out_dir / "model_checkpoint.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[checkpoint] 模型已保存: {ckpt_path}")

    def _save_metrics(self, metrics: dict, run_info: dict) -> None:
        """把本次实验的「配置 + 数据规模 + 指标」落盘为 metrics.json，
        并追加一行到 outputs/experiments_log.csv 便于多组实验横向对比。"""
        m = self.cfg.model
        record = {
            "run_id": self.out_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            # -- 实验配置（换时序模型/建图/标签变换做对比时看这里）--
            "temporal_encoder": m.temporal_encoder,
            "graph_scheme": m.graph_scheme,
            "graph_lag": m.graph_lag,
            "label_transform": m.label_transform,
            "active_labelers": list(self.cfg.labels.active_labelers),
            "hidden_dim": m.hidden_dim,
            "dropout": m.dropout,
            "temporal_kernel": m.temporal_kernel,
            "epochs": m.epochs,
            "lr": m.lr,
            "weight_decay": m.weight_decay,
            "seq_len": self.cfg.seq_len,
            # -- 数据规模 --
            **run_info,
            # -- 测试指标 --
            **{k: v for k, v in metrics.items()},
        }
        # 1) 单次详情
        (self.out_dir / "metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        # 2) 追加到全局台账（跨 run 对比）
        ledger = self.cfg.outputs_dir / "experiments_log.csv"
        row = pd.DataFrame([record])
        if ledger.exists():
            row.to_csv(ledger, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(ledger, index=False, encoding="utf-8-sig")
        print(f"[metrics] 已写入 {self.out_dir / 'metrics.json'}；台账 {ledger}")

    def _train_loop(self, x, y, edge_index, edge_weight,
                    sample_weight: Optional[np.ndarray] = None,
                    bundle: Optional[DatasetBundle] = None,
                    eval_every: int = 0) -> dict:
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr,
                               weight_decay=self.cfg.model.weight_decay)
        losses = []
        self.model.train()
        use_weighted = sample_weight is not None
        if use_weighted:
            w_t = torch.tensor(sample_weight, dtype=torch.float, device=self.device).view(-1, 1)

        best_metric = -np.inf
        best_epoch = 0
        best_state = None
        eval_log = []

        for epoch in range(self.cfg.model.epochs):
            opt.zero_grad()
            out = self.model(x, edge_index, edge_weight)
            if use_weighted:
                per_sample = (out - y).pow(2)
                loss = (per_sample * w_t).mean()
            else:
                loss = nn.functional.mse_loss(out, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

            if epoch % 50 == 0:
                tag = "R2" if use_weighted else ""
                print(f"[train{tag}] epoch {epoch}: loss={loss.item():.6f}")

            if eval_every > 0 and (epoch + 1) % eval_every == 0 and bundle is not None and len(bundle.test.X) > 0:
                self.model.eval()
                metrics, _ = self._evaluate(bundle)
                self.model.train()
                # 用 R²(prob) 作为选择最佳 epoch 的指标
                score = metrics.get('r2', -np.inf)
                eval_log.append({'epoch': epoch + 1, **metrics})
                print(f"[eval@{epoch+1}] R²={score:.4f} Spearman={metrics.get('spearman',0):.4f} IC={metrics.get('ic',0):.4f}")
                if score > best_metric:
                    best_metric = score
                    best_epoch = epoch + 1
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        result = {'losses': losses, 'eval_log': eval_log}
        if best_state is not None:
            result['best_epoch'] = best_epoch
            result['best_state'] = best_state
            result['best_metric'] = best_metric
        return result

    def _evaluate(self, bundle: DatasetBundle) -> tuple[dict, pd.DataFrame]:
        test = bundle.test
        n, t, f = test.X.shape
        Xte = self.scaler.transform(test.X.reshape(-1, f)).reshape(n, t, f)
        x_te = torch.tensor(Xte, dtype=torch.float, device=self.device)
        te_ei, te_ew = self._build_graph(bundle.edges, test.companies, test.pred_years)
        if te_ei is not None:
            te_ei, te_ew = te_ei.to(self.device), te_ew.to(self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(x_te, te_ei, te_ew).cpu().numpy().flatten()
        pred = self._to_prob(raw)
        actual = test.y

        mse = mean_squared_error(actual, pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        r2_logit = r2_score(_logit(actual), _logit(pred))
        spearman = _safe_corr(spearmanr, actual, pred)
        ic = _safe_corr(pearsonr, actual, pred)

        # 评级分类准确性
        rating_acc = _rating_accuracy(actual, pred)

        # 二分类判别能力：AUC / KS（违约 = default_probability >= 0.5，对应 ST/*ST 事件）
        bin_y = (actual >= 0.5).astype(int)
        n_pos = int(bin_y.sum())
        auc = None
        ks = None
        if 2 <= n_pos <= n - 2:
            auc = float(roc_auc_score(bin_y, pred))
            order = np.argsort(pred)
            cpos = np.cumsum(bin_y[order]) / n_pos
            cneg = np.cumsum(1 - bin_y[order]) / (n - n_pos)
            ks = float(np.max(np.abs(cpos - cneg)))

        print(f"[eval] 测试集 MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f} R2(prob)={r2:.4f} "
              f"R2(logit)={r2_logit:.4f} Spearman={spearman:.4f} IC(Pearson)={ic:.4f}")
        if auc is not None:
            print(f"[eval] 判别力: AUC={auc:.4f} KS={ks:.4f}（违约阈值>=0.5，正样本 {n_pos}/{n}）")
        print(f"[eval] 评级准确性: {rating_acc['rating_accuracy']:.3f} "
              f"({rating_acc['correct_predictions']}/{rating_acc['total_samples']})")
        if rating_acc['per_class']:
            for cls, stats in rating_acc['per_class'].items():
                print(f"[eval]   {cls}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['count']})")

        results = pd.DataFrame({
            "symbol": test.companies,
            "prediction_year": test.pred_years,
            "sequence_years": test.sequence_years,
            "actual_probability": actual,
            "predicted_probability": pred,
            "actual_rating": [_prob_to_rating(p) for p in actual],
            "predicted_rating": [_prob_to_rating(p) for p in pred],
        })
        results.to_csv(self.out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        self.viz.dispatch("evaluation", {"results": results}, self.out_dir)
        metrics = {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2),
                "r2_logit": float(r2_logit), "spearman": float(spearman),
                "ic": float(ic), "n_test": int(n),
                "auc": auc, "ks": ks, "n_default": n_pos,
                "rating_accuracy": float(rating_acc['rating_accuracy']),
                "rating_correct": int(rating_acc['correct_predictions']),
                "rating_total": int(rating_acc['total_samples']),
        }
        return metrics, results

    def _predict_future(self, bundle: DatasetBundle) -> None:
        fut = bundle.future
        n, t, f = fut.X.shape
        Xf = self.scaler.transform(fut.X.reshape(-1, f)).reshape(n, t, f)
        x_f = torch.tensor(Xf, dtype=torch.float, device=self.device)
        # 按预测年建图（2025），取 2025-lag 年的边结构
        pred_years = [2025] * n
        f_ei, f_ew = self._build_graph(bundle.edges, fut.companies, pred_years)
        if f_ei is not None:
            f_ei, f_ew = f_ei.to(self.device), f_ew.to(self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(x_f, f_ei, f_ew).cpu().numpy().flatten()
        pred = self._to_prob(raw)
        out = pd.DataFrame({
            "symbol": fut.companies,
            "sequence_years": fut.sequence_years,
            "prediction_year": 2025,
            "predicted_probability": pred,
        })
        out.to_csv(self.out_dir / "2025_predictions.csv", index=False, encoding="utf-8-sig")
        print(f"[predict] 2025 推演 {len(out)} 家，平均违约概率={pred.mean():.4f}")

    def _save_split_csv(self, bundle: DatasetBundle) -> None:
        for name, split in (("train", bundle.train), ("test", bundle.test)):
            if len(split.X) == 0:
                continue
            df = pd.DataFrame({
                "symbol": split.companies,
                "sequence_years": split.sequence_years,
                "prediction_year": split.pred_years,
                self.cfg.label_column: split.y,
            })
            for i in range(split.X.shape[1]):
                for j in range(split.X.shape[2]):
                    df[f"year_{i+1}_feature_{j+1}"] = split.X[:, i, j]
            df.to_csv(self.out_dir / f"{name}_dataset.csv", index=False, encoding="utf-8-sig")
