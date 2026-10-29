"""Application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.proxy import close_client
from app.registry import registry
from app.routers import benchmark, health, models, openai, pooling, rerank
from app.stats import tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("llm-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "%s v%s starting — %d model(s)",
        settings.service_name,
        settings.service_version,
        len(registry.list_all()),
    )
    for m in registry.list_all():
        logger.info("  %-25s → %s  [%s]", m.id, m.backend_url, m.model_type)
    if not settings.api_key and not settings.admin_api_key:
        logger.warning(
            "No API_KEY set — the gateway is open to anyone who can reach it"
        )
    elif not settings.admin_api_key:
        logger.warning(
            "No ADMIN_API_KEY set — every API key can add/change models and see backend URLs"
        )
    tracker.start_flush_loop()
    yield
    await tracker.save()
    await close_client()
    logger.info("%s stopped", settings.service_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        description=settings.service_description,
        root_path=settings.root_path,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(openai.router)
    app.include_router(rerank.router)  # <-- dedicated rerank + score
    app.include_router(pooling.router)  # <-- /classify + /pooling (vLLM pooling runner)
    app.include_router(benchmark.router)

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get("/")
    async def root():
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "docs": "/docs",
        }

    return app


app = create_app()
