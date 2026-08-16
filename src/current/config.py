"""集中式配置：路径、常量、超参、Tushare 凭据。

所有其他模块都从这里读取路径与常量，避免散落的硬编码。
新采集/产出数据统一落在项目根目录的 ``repository/``。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# config.py 位于 <root>/src/current/config.py，故根目录向上三级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 供应链原始 Excel（唯一的“外部原始输入”，非 Tushare 采集）。
RAW_SUPPLY_CHAIN_XLSX = PROJECT_ROOT / "data" / "raw" / "整合的供应链数据.xlsx"
RAW_SUPPLY_CHAIN_SHEET = "供应链网络数据"

# 采集与产出根目录
REPO_DIR = PROJECT_ROOT / "repository"
RAW_DIR = REPO_DIR / "raw"            # Tushare 原始缓存
CACHE_DIR = RAW_DIR / "cache"          # 单次 API 调用级缓存
INTERIM_DIR = REPO_DIR / "interim"     # 中间产物（financial/market/edges）
PROCESSED_DIR = REPO_DIR / "processed" # nodes/edges/labels parquet
OUTPUTS_DIR = REPO_DIR / "outputs"     # 训练产物

_ALL_DIRS = [REPO_DIR, RAW_DIR, CACHE_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR]

# ---------------------------------------------------------------------------
# 建模常量（与 legacy 对齐，保证同效果）
# ---------------------------------------------------------------------------
# 模型实际使用的 10 个财务特征列（顺序即张量最后一维顺序）。
FEATURE_COLUMNS = [
    "debt_to_asset_ratio", "current_ratio", "quick_ratio", "interest_coverage_ratio",
    "total_assets", "total_liab", "current_assets", "current_liab", "revenue", "operate_profit",
]

# 监督目标列（由标签模块产出，P0 为 KMV 违约概率）。
LABEL_COLUMN = "default_probability"

SEQ_LEN = 3                 # 连续 3 年特征
TRAIN_PRED_YEARS = (2007, 2020)  # 训练集：预测年份区间（含端点）
TEST_PRED_YEARS = (2021, 2024)   # 测试集：预测年份区间（含端点）
FUTURE_PREDICT_YEAR = 2025       # 用最近 3 年推演的目标年
FUTURE_INPUT_YEARS = (2022, 2023, 2024)


@dataclass
class ModelConfig:
    """TGC 模型与训练超参。"""
    temporal_encoder: str = "gated_conv"   # 时序模型插入点：见 models/temporal.py 注册名
    hidden_dim: int = 64
    dropout: float = 0.3
    temporal_kernel: int = 3
    epochs: int = 500                # 最优训练轮数（通过 0-2000 轮搜索确定）
    lr: float = 1e-3
    weight_decay: float = 1e-5
    min_edges_for_gcn: int = 10            # 样本级边数不足则退回简化卷积

    # 建图方案：
    #   "pred_year"      —— 推荐：每个样本用「预测年 - graph_lag」的供应链结构，做分年块对角图，
    #                       训练/测试口径一致，避免 legacy 的训练/测试拓扑错配。
    #   "densest_legacy" —— 已废弃：整批共用「有效边最多的一年」拓扑（保留仅作对照，见 dataset.build_graph）。
    graph_scheme: str = "pred_year"
    graph_lag: int = 1                     # 用「预测年 - lag」年的边作为该样本图结构（预测时点可得）

    # 标签变换：none | logit。logit 把挤在 0 附近的 default_probability 展开到实数域，
    # 缓解低方差导致的 R² 极不稳定；训练在 logit 空间做 MSE，评估/预测再 sigmoid 还原为概率。
    label_transform: str = "logit"


@dataclass
class TushareConfig:
    """Tushare 采集节流参数。

    按账号额度这些接口可达约 500 次/分钟。这里的限频是“全局”跨接口计数；
    财务阶段 balancesheet/income 交替、行情阶段 daily/daily_basic 交替。
    实测 480 QPM 会频繁触发账号侧限频（导致每次 60s 退避重试），
    故降为 200 QPM，单接口约 100/分钟，稳妥不频繁触发限流。
    """
    max_requests_per_minute: int = 200
    min_interval_sec: float = 0.3          # 相邻请求最小间隔（60/200=0.3）
    max_retries: int = 5
    retry_backoff_sec: float = 60.0        # 命中限频后的基础退避（按次数递增）
    request_timeout_hint: str = "tushare pro_api"
    use_cache: bool = True                 # 命中缓存则不重复消耗额度


@dataclass
class LabelConfig:
    """标签生成配置。active_labelers 决定启用哪些风险标签插入点。"""
    active_labelers: list[str] = field(default_factory=lambda: ["kmv"])
    default_point_ratio: float = 0.7       # KMV 违约点 = 总负债 × 该比例（简化版）
    min_asset_volatility: float = 0.3      # 资产波动率下限（提高到 0.3 缓解标签集中）
    fallback_volatility: float = 0.3
    risk_free_rate: float = 0.025          # 无风险利率（10 年期国债收益率，用于标准 KMV）
    kmv_time_horizon: float = 1.0          # KMV 时间 horizon T（年）
    kmv_use_iterative: bool = False        # 简化版（迭代版会压缩标签分布，效果差）


@dataclass
class VizConfig:
    """绘图数据收集插件配置。active_exporters 为已启用的插件注册名。"""
    active_exporters: list[str] = field(default_factory=lambda: ["training_curve", "scatter", "graph_snapshot"])


@dataclass
class ReflectionConfig:
    """增强点 D：反思 Agent 配置。"""
    enabled: bool = True
    error_quantile: float = 0.3
    max_weight_factor: float = 3.0
    min_weight_factor: float = 0.3
    year_weight_sensitivity: float = 0.5
    conflict_threshold: float = 0.1
    label_conflict_downweight: float = 0.5
    llm_enabled: bool = True
    label_verify_enabled: bool = True


@dataclass
class AgentConfig:
    """Agent 集成插入点配置。"""
    enabled: bool = True
    hook: str = "reflection"
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)


@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    raw_supply_chain_xlsx: Path = RAW_SUPPLY_CHAIN_XLSX
    raw_supply_chain_sheet: str = RAW_SUPPLY_CHAIN_SHEET
    repo_dir: Path = REPO_DIR
    raw_dir: Path = RAW_DIR
    cache_dir: Path = CACHE_DIR
    interim_dir: Path = INTERIM_DIR
    processed_dir: Path = PROCESSED_DIR
    outputs_dir: Path = OUTPUTS_DIR

    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    label_column: str = LABEL_COLUMN
    seq_len: int = SEQ_LEN

    model: ModelConfig = field(default_factory=ModelConfig)
    tushare: TushareConfig = field(default_factory=TushareConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # 中间/产出文件名
    financial_interim: Path = INTERIM_DIR / "financial.parquet"
    market_interim: Path = INTERIM_DIR / "market.parquet"
    edges_interim: Path = INTERIM_DIR / "edges.parquet"
    labels_interim: Path = INTERIM_DIR / "labels.parquet"
    nodes_parquet: Path = PROCESSED_DIR / "nodes.parquet"
    edges_parquet: Path = PROCESSED_DIR / "edges.parquet"
    labels_parquet: Path = PROCESSED_DIR / "labels.parquet"

    def ensure_dirs(self) -> None:
        for d in _ALL_DIRS:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# .env 加载 + 凭据
# ---------------------------------------------------------------------------
def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 解析：KEY=VALUE，不覆盖已存在的环境变量。避免额外依赖。"""
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except Exception:
        # .env 解析失败不应阻断非采集流程
        pass


def get_tushare_token() -> str:
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError(
            "未找到 TUSHARE_TOKEN。请在项目根目录 .env 写入 TUSHARE_TOKEN=... "
            "或设置同名环境变量。"
        )
    return token


def get_llm_config() -> dict:
    load_dotenv()
    return {
        "endpoint": os.getenv("SILICON_ENDPOINT", ""),
        "api_key": os.getenv("SILICON_APIKEY", ""),
        "model": os.getenv("SILICON_MODEL", ""),
    }


# 全局默认配置实例（其他模块直接 from src.current.config import CONFIG）
CONFIG = Config()
CONFIG.ensure_dirs()
