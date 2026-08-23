"""手动把供应链图导入/校验 Neo4j 的 CLI（核心逻辑见 server/services/neo4j_sync.py）。

server 启动时也会自动同步一次；本脚本用于手动全量重建或校验。

用法：
    .venv\\Scripts\\python.exe scripts/export_neo4j.py           # 同步 + 校验
    .venv\\Scripts\\python.exe scripts/export_neo4j.py --wipe    # 清空后同步

连接配置：从 .env 读取 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD（见 config.get_neo4j_config）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from server.services.neo4j_sync import sync_neo4j


def main() -> int:
    parser = argparse.ArgumentParser(description="供应链图导入 Neo4j")
    parser.add_argument("--wipe", action="store_true", help="导入前清空全图")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    result = sync_neo4j(wipe=args.wipe, batch_size=args.batch_size)

    if result["status"] == "ok":
        flag = "一致 ✅" if result["consistent"] else "不一致 ❌"
        print(f"[neo4j] 同步完成: db={result['database']} 节点={result['nodes']} "
              f"关系={result['relationships']}（parquet 边 {result['parquet_edges']}，{flag}）")
        return 0 if result["consistent"] else 1
    if result["status"] == "skipped":
        print(f"[neo4j] 跳过: {result['reason']}")
        return 0
    print(f"[neo4j] 同步失败({result['status']}): {result.get('reason')}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
