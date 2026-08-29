"""FastAPI 数据服务（读 repository/ 产物，供前端可视化）。

启动：
    .venv\\Scripts\\python.exe server/main.py            # 默认 127.0.0.1:8000
    .venv\\Scripts\\python.exe server/main.py --port 8000

接口清单见 docs/前端开发计划.md §9。所有数据直读 parquet/csv，
按文件 mtime 缓存，重训后无需重启服务。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 控制台/日志统一 UTF-8，避免 Windows GBK 无法编码中文/emoji
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routers import graph, model, company

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "current" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时同步预加载所有年度图进缓存（带日志埋点，见 services/graph_cache）。

    Neo4j 依赖已移除：图数据全部来自 processed parquet，启动慢一点换取查询毫秒级。
    """
    from server.services.graph_cache import preload_all_graphs

    preload_all_graphs()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="供应链风险评估 API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(graph.router, prefix="/api")
    app.include_router(model.router, prefix="/api")
    app.include_router(company.router, prefix="/api")

    @app.get("/api/health")
    def health():
        from server.services.graph_cache import cache_stats

        return {"status": "ok", **cache_stats()}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """生产模式：托管前端构建产物（frontend/current/dist），含 SPA 回退。

    仅当 dist 存在时生效；开发模式用 vite dev server（:5173）即可。
    /api 路由优先于下面的兜底路由，互不冲突。
    """
    if not FRONTEND_DIST.exists():
        return
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="供应链风险评估数据服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
