from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from bad_debt_app.api.config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_SNAPSHOT_DATE,
    MODEL_REGISTRY,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import model_public_info

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_KEY,
        "snapshot_date": DEFAULT_SNAPSHOT_DATE,
        "thresholds": {"low": THRESHOLD_LOW, "high": THRESHOLD_HIGH},
        "models": [model_public_info(k, v) for k, v in MODEL_REGISTRY.items()],
    }


@router.get("/", response_class=HTMLResponse)
def ui_index():
    return (
        '<!doctype html><html><head><meta http-equiv="refresh" content="0;url=/web/" /></head>'
        '<body><p>Redirecting to <a href="/web/">/web/</a>...</p></body></html>'
    )
