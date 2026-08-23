"""命令行入口：python -m src.current.cli {collect|edges|events|label|export|train|predict|all}

示例：
  python -m src.current.cli all              # 采集->标签->导出->训练 全流程
  python -m src.current.cli edges            # 仅从本地 Excel 重建边（不耗额度）
  python -m src.current.cli collect          # 采集财务+行情+风险事件（消耗 Tushare 额度）
  python -m src.current.cli events           # 仅采集风险事件（ST 状态/退市，方案D 真值）
  python -m src.current.cli label            # 仅生成标签（默认 kmv 基线，--scheme hybrid 混合）
  python -m src.current.cli export           # 仅导出三张 parquet
  python -m src.current.cli train            # 仅训练+评估+2025推演（需已导出）
"""
from __future__ import annotations

import argparse
import sys

# 控制台统一 UTF-8，避免 Windows GBK 无法编码中文/R² 中断
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.current import pipeline


def _redirect_logs(path: str) -> None:
    """让 Python 直接以 UTF-8 写日志文件，绕开所有 shell/控制台编码问题。"""
    f = open(path, "w", encoding="utf-8", buffering=1)  # 行缓冲，实时可读
    sys.stdout = f
    sys.stderr = f


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="供应链金融风控 current 流水线")
    parser.add_argument("command",
                        choices=["collect", "edges", "events", "label", "export", "train", "predict", "all"],
                        help="要执行的阶段")
    parser.add_argument("--no-resume", action="store_true", help="采集时不使用断点续跑")
    parser.add_argument("--scheme", metavar="NAME", default=None,
                        help="标签方案（label 阶段，如 kmv/hybrid；默认取 config）")
    parser.add_argument("--log", metavar="FILE", default=None,
                        help="把 stdout/stderr 以 UTF-8 写入该文件（推荐后台运行时使用）")
    args = parser.parse_args(argv)

    if args.log:
        _redirect_logs(args.log)

    resume = not args.no_resume
    cmd = args.command

    if cmd == "collect":
        pipeline.step_collect(resume=resume)
    elif cmd == "edges":
        pipeline.step_edges_only()
    elif cmd == "events":
        pipeline.step_events(resume=resume)
    elif cmd == "label":
        pipeline.step_label(scheme=args.scheme)
    elif cmd == "export":
        pipeline.step_export()
    elif cmd in ("train", "predict"):
        # train 已包含评估与 2025 推演
        result = pipeline.step_train()
        print(f"完成: {result}")
    elif cmd == "all":
        result = pipeline.run_all(resume=resume)
        print(f"完成: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
