from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backend.api import get_dashboard_payload_service, router
from dashboard.backend.services.dashboard_payload import DashboardPayloadService


def get_frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app(
    *,
    default_selected_run_ids: list[str] | None = None,
    frontend_dist: Path | None = None,
    dashboard_payload_service: DashboardPayloadService | None = None,
) -> FastAPI:
    app = FastAPI(title="Dashboard")
    app.state.default_selected_run_ids = list(default_selected_run_ids or [])
    if dashboard_payload_service is not None:
        app.dependency_overrides[get_dashboard_payload_service] = lambda: dashboard_payload_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    using_explicit_dist = frontend_dist is not None
    dist_dir = (frontend_dist or get_frontend_dist_dir()).resolve()
    if using_explicit_dist:
        index_path = dist_dir / "index.html"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing frontend entrypoint: {index_path}")

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="dashboard-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str) -> FileResponse:
        if path.startswith("api"):
            raise HTTPException(status_code=404, detail="not found")

        if not dist_dir.exists():
            raise HTTPException(status_code=503, detail="dashboard frontend is not built")

        index_path = dist_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=503, detail="dashboard frontend is not built")

        if path:
            candidate = (dist_dir / path).resolve(strict=False)
            try:
                candidate.relative_to(dist_dir)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="not found") from exc

            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)

        return FileResponse(index_path)

    return app


app = create_app()
