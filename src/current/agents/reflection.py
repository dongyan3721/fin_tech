"""增强点 D：反思 Agent — 难例重加权 + LLM 诊断 + 标签交叉验证。

核心流程：
1. 对测试集预测做误差归因（按年份/财务特征/图位置分组）
2. 可选：用 Tushare 查询 ST 状态交叉验证 KMV 标签
3. 可选：用 LLM 做深度诊断，输出加权建议
4. 生成训练样本加权规则 → 下一轮加权训练
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from src.current.agents.base import AgentHook
from src.current.config import CONFIG, get_llm_config
from src.current.registry import AGENT_HOOKS
from src.current.train.dataset import DatasetBundle


@AGENT_HOOKS.register("reflection")
class ReflectionAgent(AgentHook):

    def __init__(self, config=None):
        self.cfg = config or CONFIG.agent.reflection
        self._llm_cfg = get_llm_config()
        self._report: dict = {}
        self._st_records: Optional[pd.DataFrame] = None

    # ================================================================
    # 1. 统计分析
    # ================================================================
    def analyze_errors(
        self,
        test_preds: pd.DataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict:
        pred = test_preds["predicted_probability"].values
        actual = test_preds["actual_probability"].values
        error = pred - actual
        ae = np.abs(error)

        summary = {
            "n_samples": len(test_preds),
            "mean_abs_error": float(ae.mean()),
            "median_abs_error": float(np.median(ae)),
            "std_abs_error": float(ae.std()),
            "max_abs_error": float(ae.max()),
            "mean_signed_error": float(error.mean()),
        }

        year_stats = {}
        for year, grp in test_preds.groupby("prediction_year"):
            ga = grp["actual_probability"].values
            gp = grp["predicted_probability"].values
            ge = gp - ga
            gae = np.abs(ge)
            year_stats[str(int(year))] = {
                "n": int(len(grp)),
                "mean_ae": float(gae.mean()),
                "median_ae": float(np.median(gae)),
                "signed_error": float(ge.mean()),
                "std_ae": float(gae.std()),
            }
        summary["by_year"] = year_stats

        summary["by_feature"] = self._group_by_feature(test_preds, nodes, feature_columns)

        degree = self._compute_node_degree(nodes, edges)
        test_degrees = []
        for sym in test_preds["symbol"].values:
            sym_str = str(sym).zfill(6)
            test_degrees.append(degree.get(sym_str, 0))
        test_preds = test_preds.copy()
        test_preds["degree"] = test_degrees
        summary["by_degree"] = self._group_by_metric(test_preds, ae, "degree", [0, 1, 2, 5])

        high_risk = actual > np.percentile(actual, 75)
        low_risk = actual <= np.percentile(actual, 75)
        summary["high_risk_recall"] = float(1 - (pred[high_risk] < actual[high_risk]).mean()) if high_risk.any() else None
        summary["low_risk_precision"] = float((pred[low_risk] < actual[low_risk]).mean()) if low_risk.any() else None

        self._report["error_summary"] = summary
        return summary

    def _group_by_feature(
        self, test_preds: pd.DataFrame, nodes: pd.DataFrame, feature_columns: list[str]
    ) -> dict:
        result = {}
        for feat in ["debt_to_asset_ratio", "current_ratio"]:
            if feat not in nodes.columns:
                continue
            merged = test_preds.merge(nodes[["symbol", "year", feat]], left_on=["symbol", "prediction_year"], right_on=["symbol", "year"], how="left")
            merged = merged.dropna(subset=[feat])
            if len(merged) < 10:
                continue
            med = merged[feat].median()
            result[feat] = {
                "median": float(med),
                "below_median_ae": float(np.abs(merged[merged[feat] <= med]["predicted_probability"] - merged[merged[feat] <= med]["actual_probability"]).mean()),
                "above_median_ae": float(np.abs(merged[merged[feat] > med]["predicted_probability"] - merged[merged[feat] > med]["actual_probability"]).mean()),
            }
        return result

    def _group_by_metric(self, df: pd.DataFrame, ae: np.ndarray, col: str, bins: list) -> dict:
        result = {}
        labels = []
        for i in range(len(bins)):
            if i == 0:
                labels.append(f"<{bins[i]}")
            else:
                labels.append(f"{bins[i-1]}-{bins[i]}")
        labels.append(f">={bins[-1]}")
        all_bins = [-np.inf] + bins + [np.inf]
        df = df.reset_index(drop=True)
        ae = np.asarray(ae).flatten()
        if len(ae) != len(df):
            return result
        groups = pd.cut(df[col], bins=all_bins, labels=labels)
        for label in labels:
            mask = groups == label
            grp_ae = ae[mask.values]
            if len(grp_ae) == 0:
                continue
            result[str(label)] = {
                "n": int(len(grp_ae)),
                "mean_ae": float(grp_ae.mean()),
            }
        return result

    def _compute_node_degree(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
        degree = {}
        if edges is not None and not edges.empty:
            for sym in nodes["symbol"].unique():
                sym_str = str(sym)
                count = ((edges["source"] == sym_str) | (edges["target"] == sym_str)).sum()
                degree[sym_str] = int(count)
        return degree

    # ================================================================
    # 2. 标签交叉验证（Tushare ST 状态）
    # ================================================================
    def verify_labels(self, test_preds: pd.DataFrame) -> dict:
        if not self.cfg.label_verify_enabled:
            return {"enabled": False}

        try:
            from src.current.data.tushare_client import TushareClient
            from src.current.transform.symbols import to_ts_code

            client = TushareClient()
            st_records = []

            symbols = test_preds["symbol"].unique()
            for sym in symbols:
                ts_code = to_ts_code(str(sym).zfill(6))
                if not ts_code:
                    continue
                try:
                    df = client.query("namechange", ts_code=ts_code, fields="ts_code,start_date,end_date,ann_reason")
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            reason = str(row.get("ann_reason", ""))
                            if "ST" in reason or "st" in reason.lower():
                                st_records.append({
                                    "symbol": str(sym).zfill(6),
                                    "start_date": str(row.get("start_date", "")),
                                    "end_date": str(row.get("end_date", "")),
                                })
                except Exception:
                    continue

            if not st_records:
                return {"enabled": True, "st_count": 0, "conflicts": []}

            st_df = pd.DataFrame(st_records)
            conflicts = []
            for _, row in st_df.iterrows():
                sym = row["symbol"]
                pred_row = test_preds[test_preds["symbol"].astype(str).str.zfill(6) == sym]
                if pred_row.empty:
                    continue
                actual_prob = pred_row["actual_probability"].values[0]
                pred_prob = pred_row["predicted_probability"].values[0]
                if pred_prob < 0.05 and actual_prob > 0.05:
                    conflicts.append({
                        "symbol": sym,
                        "predicted_probability": float(pred_prob),
                        "actual_probability": float(actual_prob),
                        "issue": "ST 但 KMV 预测低风险",
                    })

            result = {
                "enabled": True,
                "st_count": len(st_df),
                "n_conflicts": len(conflicts),
                "conflicts": conflicts[:10],
            }
            self._report["label_verification"] = result
            return result

        except Exception as e:
            return {"enabled": True, "error": str(e)}

    # ================================================================
    # 3. LLM 诊断
    # ================================================================
    def llm_diagnose(self, error_summary: dict, label_verify: dict) -> dict:
        if not self.cfg.llm_enabled:
            return {"enabled": False}

        endpoint = self._llm_cfg.get("endpoint", "")
        api_key = self._llm_cfg.get("api_key", "")
        model = self._llm_cfg.get("model", "")

        if not all([endpoint, api_key, model]):
            print("[reflection] LLM 配置缺失，跳过诊断")
            return {"enabled": False, "reason": "配置缺失"}

        try:
            from openai import OpenAI

            client = OpenAI(base_url=endpoint, api_key=api_key)

            summary_text = json.dumps(error_summary, indent=2, ensure_ascii=False, default=str)
            verify_text = json.dumps(label_verify, indent=2, ensure_ascii=False, default=str)

            prompt = f"""你是一个供应链金融风控专家。以下是TGC时序图卷积模型在测试集上的误差分析结果。

## 误差统计
```json
{summary_text}
```

## 标签验证结果
```json
{verify_text}
```

请分析：
1. 哪些年份/特征区间的误差最大？为什么？
2. 标签验证中发现了哪些冲突？这些冲突说明什么？
3. 建议对哪些类型的样本提高/降低训练权重？

请用JSON格式返回，包含以下字段：
```json
{{
  "analysis": "你的分析（100字内）",
  "year_rules": {{"年份": 权重倍数}},
  "feature_rules": {{"特征描述": 权重倍数}},
  "confidence": 0.0-1.0
}}
```
只返回JSON，不要其他内容。"""

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content.strip()

            if not content:
                print("[reflection] LLM 返回为空，跳过诊断")
                return {"enabled": True, "error": "LLM 返回为空"}

            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                content = json_match.group()
            else:
                print("[reflection] LLM 返回中未找到 JSON，跳过诊断")
                return {"enabled": True, "error": "未找到 JSON", "raw": content[:200]}

            rules = json.loads(content)
            self._report["llm_diagnosis"] = rules
            return rules

        except Exception as e:
            print(f"[reflection] LLM 诊断失败: {e}")
            return {"enabled": True, "error": str(e)}

    # ================================================================
    # 4. 样本权重计算
    # ================================================================
    def compute_sample_weights(self, bundle: DatasetBundle) -> np.ndarray:
        test_preds = self._report.get("test_predictions")
        if test_preds is None:
            return np.ones(len(bundle.train.X))

        train = bundle.train
        n_train = len(train.X)
        weights = np.ones(n_train, dtype=np.float64)

        try:
            ae_test = np.abs(test_preds["predicted_probability"].values - test_preds["actual_probability"].values)
            test_years = test_preds["prediction_year"].values
            overall_ae = ae_test.mean()

            train_years = np.array(train.pred_years)
            unique_train_years = np.unique(train_years)

            year_error_test = {}
            for year in np.unique(test_years):
                mask = test_years == year
                year_error_test[int(year)] = float(ae_test[mask].mean())

            train_year_weights = {}
            recent_test_years = sorted([y for y in year_error_test.keys() if y >= 2021])
            if len(recent_test_years) >= 2:
                recent_ae = np.mean([year_error_test[y] for y in recent_test_years])
                early_years = [y for y in year_error_test if y < 2021]
                if early_years:
                    early_ae = np.mean([year_error_test[y] for y in early_years])
                else:
                    early_ae = overall_ae
                if recent_ae > early_ae * 1.1:
                    for year in unique_train_years:
                        if year >= 2017:
                            train_year_weights[year] = min(1.0 + (recent_ae / max(early_ae, 1e-6) - 1.0) * self.cfg.year_weight_sensitivity, self.cfg.max_weight_factor)

            for yr, high_risk_years in [(2021, [2019, 2020]), (2022, [2020]), (2023, []), (2024, [])]:
                if yr in year_error_test and year_error_test[yr] > overall_ae * 1.2:
                    factor = min(1.0 + (year_error_test[yr] / overall_ae - 1.0) * 0.3, self.cfg.max_weight_factor)
                    for hy in high_risk_years:
                        if hy in train_year_weights:
                            train_year_weights[hy] = max(train_year_weights[hy], factor)
                        else:
                            train_year_weights[hy] = factor

            for i, year in enumerate(train.pred_years):
                if year in train_year_weights:
                    weights[i] *= train_year_weights[year]

            feat_idx = {}
            for idx, col in enumerate(bundle.feature_columns):
                feat_idx[col] = idx

            if "debt_to_asset_ratio" in feat_idx:
                fi = feat_idx["debt_to_asset_ratio"]
                train_debt = train.X[:, -1, fi]
                median_debt = np.median(train_debt)
                high_debt_mask = train_debt > median_debt

                for i in range(n_train):
                    if high_debt_mask[i]:
                        weights[i] *= 1.2

            if "current_ratio" in feat_idx:
                fi = feat_idx["current_ratio"]
                train_cr = train.X[:, -1, fi]
                median_cr = np.median(train_cr)
                low_cr_mask = train_cr < median_cr

                for i in range(n_train):
                    if low_cr_mask[i]:
                        weights[i] *= 1.15

            label_verify = self._report.get("label_verification", {})
            if isinstance(label_verify, dict) and label_verify.get("conflicts"):
                conflict_symbols = {c["symbol"] for c in label_verify["conflicts"]}
                for i, sym in enumerate(train.companies):
                    sym_str = str(sym).zfill(6)
                    if sym_str in conflict_symbols:
                        weights[i] *= self.cfg.label_conflict_downweight

            llm_rules = self._report.get("llm_diagnosis", {})
            if isinstance(llm_rules, dict) and "year_rules" in llm_rules:
                for year_str, factor in llm_rules["year_rules"].items():
                    try:
                        year_int = int(year_str)
                        factor = float(factor)
                        factor = np.clip(factor, self.cfg.min_weight_factor, self.cfg.max_weight_factor)
                        for i, year in enumerate(train.pred_years):
                            if year == year_int:
                                weights[i] *= factor
                    except (ValueError, TypeError):
                        continue

        except Exception as e:
            print(f"[reflection] 权重计算异常: {e}，回退到均匀权重")
            import traceback
            traceback.print_exc()

        weights = np.clip(weights, self.cfg.min_weight_factor, self.cfg.max_weight_factor)
        weights = weights / weights.mean()

        self._report["weight_stats"] = {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "n_boosted": int((weights > 1.1).sum()),
            "n_reduced": int((weights < 0.9).sum()),
        }

        return weights

    # ================================================================
    # 5. 主入口
    # ================================================================
    def reflect(
        self,
        test_preds: pd.DataFrame,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        feature_columns: list[str],
        bundle: DatasetBundle,
    ) -> dict:
        self._report = {}
        self._report["test_predictions"] = test_preds

        print("[reflection] 误差分析...")
        error_summary = self.analyze_errors(test_preds, nodes, edges, feature_columns)

        print("[reflection] 标签交叉验证...")
        label_verify = self.verify_labels(test_preds)

        print("[reflection] LLM 诊断...")
        llm_diag = self.llm_diagnose(error_summary, label_verify)

        print("[reflection] 计算样本权重...")
        weights = self.compute_sample_weights(bundle)

        ws = self._report.get("weight_stats", {})
        print(f"[reflection] 权重统计: mean={ws.get('mean', 1.0):.3f}, "
              f"boosted={ws.get('n_boosted', 0)}, "
              f"reduced={ws.get('n_reduced', 0)}")

        return {
            "weights": weights,
            "report": self._report,
            "metrics": {
                "mean_abs_error": error_summary["mean_abs_error"],
                "mean_signed_error": error_summary["mean_signed_error"],
            },
        }
