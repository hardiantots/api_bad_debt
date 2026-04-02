from __future__ import annotations

import logging
import secrets
import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from bad_debt_app.api.config import ALLOW_ORIGINS, API_KEY, APP_TITLE, BASE_DIR
from bad_debt_app.api.routes_db import router as db_router
from bad_debt_app.api.routes_system import router as system_router
from bad_debt_app.api.routes_upload import router as upload_router

logger = logging.getLogger("bad_debt_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

web_dir = BASE_DIR / "web"
if web_dir.is_dir():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")


@app.middleware("http")
async def add_timing(request: Request, call_next):
    if API_KEY:
        public_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/"}
        if (
            not request.url.path.startswith("/web")
            and request.url.path not in public_paths
        ):
            req_api_key = (
                request.headers.get("x-api-key")
                or request.headers.get("authorization", "")
                .removeprefix("Bearer ")
                .strip()
            )
            if not req_api_key or not secrets.compare_digest(req_api_key, API_KEY):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid or missing API key."},
                )

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    logger.info(
        "%s %s -> %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


app.include_router(system_router)
app.include_router(upload_router)
app.include_router(db_router)
