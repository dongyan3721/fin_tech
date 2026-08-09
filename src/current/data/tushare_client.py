"""全新 Tushare 客户端封装（不依赖 legacy 采集代码）。

特性：
- 令牌桶式限频（滑动窗口 max_requests_per_minute + 相邻请求最小间隔）；
- 命中限频/网络错误自动退避重试；
- 单次调用级磁盘缓存（parquet），命中缓存不消耗额度，天然支持断点续跑。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.current.config import CONFIG, TushareConfig, get_tushare_token


class RateLimiter:
    """滑动窗口限频 + 相邻请求最小间隔。"""

    def __init__(self, max_per_minute: int, min_interval: float) -> None:
        self.max_per_minute = max_per_minute
        self.min_interval = min_interval
        self._timestamps: list[float] = []
        self._last = 0.0

    def acquire(self) -> None:
        now = time.time()
        # 清理 60s 之前的时间戳
        self._timestamps = [t for t in self._timestamps if t > now - 60]
        if len(self._timestamps) >= self.max_per_minute:
            wait = 60 - (now - min(self._timestamps))
            if wait > 0:
                time.sleep(wait)
                now = time.time()
                self._timestamps = [t for t in self._timestamps if t > now - 60]
        # 相邻间隔
        gap = now - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.time()
        self._timestamps.append(self._last)


class TushareClient:
    def __init__(self, cfg: Optional[TushareConfig] = None) -> None:
        self.cfg = cfg or CONFIG.tushare
        self._limiter = RateLimiter(self.cfg.max_requests_per_minute, self.cfg.min_interval_sec)
        self._pro = None  # 延迟初始化，避免无 token 时导入即失败

    # -- 底层 -------------------------------------------------------------
    def _ensure_api(self):
        if self._pro is None:
            import tushare as ts  # 延迟导入
            ts.set_token(get_tushare_token())
            self._pro = ts.pro_api()
        return self._pro

    def _cache_path(self, api_name: str, params: dict) -> Path:
        key = json.dumps({"api": api_name, **params}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        d = CONFIG.cache_dir / api_name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}.parquet"

    def query(self, api_name: str, **params) -> pd.DataFrame:
        """调用任意 Tushare pro 接口，带缓存/限频/重试。返回 DataFrame（可能为空）。"""
        cache_file = self._cache_path(api_name, params)
        if self.cfg.use_cache and cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass  # 缓存损坏则重新拉取

        pro = self._ensure_api()
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                self._limiter.acquire()
                func = getattr(pro, api_name)
                df = func(**params)
                if df is None:
                    df = pd.DataFrame()
                if self.cfg.use_cache:
                    try:
                        df.to_parquet(cache_file, index=False)
                    except Exception:
                        pass
                return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                if any(k in msg for k in ("每分钟", "频率", "rate limit", "IP", "超限")):
                    wait = self.cfg.retry_backoff_sec * (attempt + 1)
                    print(f"[tushare] 限频，{wait:.0f}s 后重试 ({attempt+1}/{self.cfg.max_retries}): {api_name}")
                    time.sleep(wait)
                elif any(k in msg.lower() for k in ("connection", "timeout", "网络")):
                    time.sleep(3 * (attempt + 1))
                else:
                    print(f"[tushare] 调用 {api_name} 失败: {msg}")
                    time.sleep(2)
        print(f"[tushare] {api_name} 达最大重试次数，返回空。最后错误: {last_err}")
        return pd.DataFrame()

    # -- 便捷封装 ---------------------------------------------------------
    def balancesheet(self, ts_code: str, period: str, fields: str) -> pd.DataFrame:
        return self.query("balancesheet", ts_code=ts_code, period=period, fields=fields)

    def income(self, ts_code: str, period: str, fields: str) -> pd.DataFrame:
        return self.query("income", ts_code=ts_code, period=period, fields=fields)

    def daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.query("daily", ts_code=ts_code, start_date=start_date, end_date=end_date)

    def daily_basic(self, ts_code: str, start_date: str, end_date: str, fields: str) -> pd.DataFrame:
        return self.query("daily_basic", ts_code=ts_code, start_date=start_date,
                          end_date=end_date, fields=fields)
