"""生成行业—商品期货映射表与权重表（市场风险标签手册 §5，LLM 规则化评分）。

RelationScore = 0.4×原材料 + 0.3×产品 + 0.2×成本 + 0.1×产业链
筛选：score >= 0.3，每行业按分数降序保留 Top5；权重 = score / Σscore。
遵守 §5.3 特殊规则：不因"常见工业品"硬凑关系（金融/地产/医药等无直接商品关联 → 不映射）。

输出：repository/market/industry_commodity_mapping.csv / industry_commodity_weight.csv
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(r"D:\Data\CodeRepository\py\GraghRiskEvaluate")
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "repository" / "market"

# 商品期货（Tushare 主力连续代码）：code -> (名称, 交易所后缀)
COMMODITIES = {
    "CU": ("铜", "SHF"), "AL": ("铝", "SHF"), "ZN": ("锌", "SHF"), "NI": ("镍", "SHF"),
    "PB": ("铅", "SHF"), "SN": ("锡", "SHF"), "AU": ("黄金", "SHF"), "AG": ("白银", "SHF"),
    "I": ("铁矿石", "DCE"), "JM": ("焦煤", "DCE"), "J": ("焦炭", "DCE"),
    "RB": ("螺纹钢", "SHF"), "HC": ("热轧卷板", "SHF"),
    "ZC": ("动力煤", "ZCE"),
    "SC": ("原油", "INE"), "FU": ("燃料油", "SHF"), "BU": ("沥青", "SHF"),
    "L": ("塑料", "DCE"), "PP": ("聚丙烯", "DCE"),
    "TA": ("PTA", "ZCE"), "MA": ("甲醇", "ZCE"), "EG": ("乙二醇", "DCE"),
    "SA": ("纯碱", "ZCE"), "RU": ("天然橡胶", "SHF"), "V": ("PVC", "DCE"), "FG": ("玻璃", "ZCE"),
    "CF": ("棉花", "ZCE"), "PF": ("短纤", "ZCE"), "SR": ("白糖", "ZCE"),
    "C": ("玉米", "DCE"), "A": ("大豆", "DCE"), "M": ("豆粕", "DCE"), "Y": ("豆油", "DCE"), "P": ("棕榈油", "DCE"),
    "JD": ("鸡蛋", "DCE"), "AP": ("苹果", "ZCE"),
    "SP": ("纸浆", "SHF"),
    "SI": ("工业硅", "GFE"), "LC": ("碳酸锂", "GFE"),
}

# 行业 -> [(商品, 原材料, 产品, 成本, 产业链, 关系类型)]，按手册四维度打分
# 类型: upstream_material / production_input / downstream_product / energy_cost
INDUSTRY_RELATIONS = {
    "801050.SI": ("有色金属", [
        ("CU", 1, 1, 0, 1, "downstream_product"), ("AL", 1, 1, 0, 1, "downstream_product"),
        ("ZN", 1, 1, 0, 1, "downstream_product"), ("NI", 1, 1, 0, 1, "downstream_product"),
        ("PB", 1, 1, 0, 0, "downstream_product"), ("SN", 1, 1, 0, 0, "downstream_product"),
        ("AU", 0, 1, 0, 0, "downstream_product"), ("AG", 0, 1, 0, 0, "downstream_product"),
    ]),
    "801040.SI": ("钢铁", [
        ("I", 1, 0, 1, 1, "upstream_material"), ("JM", 1, 0, 1, 1, "upstream_material"),
        ("J", 0, 0, 1, 1, "upstream_material"), ("RB", 0, 1, 0, 1, "downstream_product"),
        ("HC", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801950.SI": ("煤炭", [
        ("ZC", 0, 1, 0, 1, "downstream_product"), ("JM", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801960.SI": ("石油石化", [
        ("SC", 0, 1, 0, 1, "downstream_product"), ("FU", 0, 1, 0, 1, "downstream_product"),
        ("BU", 0, 1, 0, 1, "downstream_product"), ("L", 0, 1, 0, 1, "downstream_product"),
        ("PP", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801030.SI": ("基础化工", [
        ("TA", 0, 1, 0, 1, "downstream_product"), ("MA", 0, 1, 0, 1, "downstream_product"),
        ("EG", 0, 1, 0, 1, "downstream_product"), ("SA", 0, 1, 0, 1, "downstream_product"),
        ("RU", 0, 1, 0, 1, "downstream_product"), ("V", 0, 1, 0, 1, "downstream_product"),
        ("SC", 0, 0, 1, 1, "energy_cost"),
    ]),
    "801130.SI": ("纺织服饰", [
        ("CF", 1, 0, 0, 1, "upstream_material"), ("TA", 1, 0, 0, 1, "upstream_material"),
        ("PF", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801010.SI": ("农林牧渔", [
        ("C", 0, 1, 0, 1, "downstream_product"), ("A", 0, 1, 0, 1, "downstream_product"),
        ("M", 0, 1, 0, 1, "downstream_product"), ("Y", 0, 1, 0, 1, "downstream_product"),
        ("P", 0, 1, 0, 1, "downstream_product"), ("SR", 0, 1, 0, 1, "downstream_product"),
        ("JD", 0, 1, 0, 1, "downstream_product"), ("AP", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801120.SI": ("食品饮料", [
        ("SR", 1, 0, 0, 1, "upstream_material"), ("Y", 1, 0, 0, 1, "upstream_material"),
        ("P", 1, 0, 0, 1, "upstream_material"), ("C", 0, 0, 1, 1, "production_input"),
    ]),
    "801110.SI": ("家用电器", [
        ("CU", 1, 0, 1, 1, "upstream_material"), ("AL", 1, 0, 0, 1, "upstream_material"),
        ("PP", 1, 0, 0, 1, "production_input"),
    ]),
    "801730.SI": ("电力设备", [
        ("CU", 1, 0, 1, 1, "upstream_material"), ("AL", 1, 0, 0, 1, "upstream_material"),
        ("NI", 1, 0, 0, 1, "upstream_material"), ("SI", 0, 1, 0, 1, "downstream_product"),
        ("LC", 0, 1, 0, 1, "downstream_product"),
    ]),
    "801890.SI": ("机械设备", [
        ("HC", 1, 0, 1, 1, "upstream_material"), ("RB", 1, 0, 0, 1, "upstream_material"),
        ("CU", 1, 0, 0, 1, "upstream_material"), ("AL", 1, 0, 0, 1, "upstream_material"),
    ]),
    "801880.SI": ("汽车", [
        ("HC", 1, 0, 1, 1, "upstream_material"), ("CU", 1, 0, 0, 1, "upstream_material"),
        ("AL", 1, 0, 0, 1, "upstream_material"), ("RU", 1, 0, 0, 1, "production_input"),
    ]),
    "801080.SI": ("电子", [
        ("CU", 1, 0, 0, 1, "upstream_material"), ("AG", 1, 0, 0, 0, "production_input"),
    ]),
    "801770.SI": ("通信", [("CU", 1, 0, 0, 1, "upstream_material")]),
    "801740.SI": ("国防军工", [("AL", 1, 0, 0, 1, "upstream_material")]),
    "801720.SI": ("建筑装饰", [("RB", 1, 0, 1, 1, "upstream_material")]),
    "801710.SI": ("建筑材料", [
        ("SA", 1, 0, 0, 1, "upstream_material"), ("FG", 0, 1, 0, 1, "downstream_product"),
        ("ZC", 0, 0, 1, 1, "energy_cost"),
    ]),
    "801160.SI": ("公用事业", [
        ("ZC", 0, 0, 1, 1, "energy_cost"), ("SC", 0, 0, 1, 1, "energy_cost"),
    ]),
    "801170.SI": ("交通运输", [
        ("FU", 0, 0, 1, 1, "energy_cost"), ("SC", 0, 0, 1, 1, "energy_cost"),
    ]),
    "801140.SI": ("轻工制造", [("SP", 1, 0, 0, 1, "upstream_material")]),
    # 无直接商品关联，按 §5.3 规则不硬凑：医药生物/银行/非银金融/房地产/计算机/传媒/
    # 商贸零售/社会服务/环保/美容护理/综合/电子中无强关联部分
}

W = {"material": 0.4, "product": 0.3, "cost": 0.2, "chain": 0.1}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    for ind_code, (ind_name, rels) in INDUSTRY_RELATIONS.items():
        scored = []
        for code, m, p, c, s, rtype in rels:
            score = round(W["material"] * m + W["product"] * p + W["cost"] * c + W["chain"] * s, 2)
            if score < 0.3:  # §5.2.3 None 不保留
                continue
            scored.append((code, score, rtype))
        # §5.2.4 Top5
        scored.sort(key=lambda x: (-x[1], x[0]))
        kept = scored[:5]
        for code, score, rtype in kept:
            level = "High" if score >= 0.6 else "Medium"
            cname = COMMODITIES[code][0]
            mapping_rows.append({
                "industry_code": ind_code, "industry_name": ind_name,
                "commodity_code": code, "commodity_name": cname,
                "relation_type": rtype, "relation_score": score,
                "relation_level": level,
            })

    # 权重归一化 + 校验
    weight_rows = []
    by_ind: dict[str, list] = {}
    for r in mapping_rows:
        by_ind.setdefault(r["industry_code"], []).append(r)
    for ind_code, rows in by_ind.items():
        total = sum(r["relation_score"] for r in rows)
        for r in rows:
            weight_rows.append({
                "industry_code": ind_code, "industry_name": r["industry_name"],
                "commodity_code": r["commodity_code"], "commodity_name": r["commodity_name"],
                "suffix": COMMODITIES[r["commodity_code"]][1],
                "relation_score": r["relation_score"],
                "weight": round(r["relation_score"] / total, 6),
                "weight_method": "relation_score_normalization",
            })

    with io.open(OUT_DIR / "industry_commodity_mapping.csv", "w", encoding="utf-8-sig", newline="") as f:
        import csv
        wcsv = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(mapping_rows)
    with io.open(OUT_DIR / "industry_commodity_weight.csv", "w", encoding="utf-8-sig", newline="") as f:
        import csv
        wcsv = csv.DictWriter(f, fieldnames=list(weight_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(weight_rows)

    n_high = sum(1 for r in mapping_rows if r["relation_level"] == "High")
    n_med = len(mapping_rows) - n_high
    used = sorted({r["commodity_code"] for r in mapping_rows})
    lines = [
        "行业—商品映射质量报告（LLM 规则化评分生成）",
        "=" * 46,
        f"有映射行业数: {len(by_ind)} / 全行业 31（其余无直接商品关联，按 §5.3 不硬凑）",
        f"行业—商品关系总数: {len(mapping_rows)}（High {n_high} / Medium {n_med}）",
        f"实际使用商品数: {len(used)} / 期货池 {len(COMMODITIES)}",
        f"使用商品: {','.join(used)}",
        f"平均每行业商品数: {len(mapping_rows) / len(by_ind):.2f}",
        "权重校验: 每行业 Σweight=1（归一化保证）",
    ]
    with io.open(OUT_DIR / "mapping_quality_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
