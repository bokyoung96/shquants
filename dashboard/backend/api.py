from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from dashboard.backend.schemas import DashboardPayloadModel, SessionBootstrapModel
from dashboard.backend.services.dashboard_payload import DashboardPayloadService
from dashboard.backend.services.run_index import RunIndexService

router = APIRouter(prefix="/api")


def get_run_index_service() -> RunIndexService:
    return RunIndexService()


def get_dashboard_payload_service() -> DashboardPayloadService:
    return DashboardPayloadService()


@router.get("/runs")
def list_runs() -> list[dict[str, object]]:
    return [run.model_dump() for run in get_run_index_service().list_runs()]


@router.get("/session", response_model=SessionBootstrapModel)
def get_session(request: Request) -> dict[str, object]:
    return {
        "default_selected_run_ids": list(getattr(request.app.state, "default_selected_run_ids", [])),
    }


@router.get("/dashboard", response_model=DashboardPayloadModel)
def get_dashboard(
    run_ids: Annotated[list[str], Query(alias="run_ids")],
    service: DashboardPayloadService = Depends(get_dashboard_payload_service),
) -> dict[str, object]:
    canonical_run_ids = list(dict.fromkeys(run_ids))
    return service.build(canonical_run_ids).model_dump(by_alias=True)
