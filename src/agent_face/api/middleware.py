"""
FastAPI-level middleware for error handling and request processing.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from agent_face.api.schemas import ErrorResponse


def setup_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                detail=str(exc),
                error_code="INVALID_INPUT",
            ).model_dump(),
        )

    @app.exception_handler(NotImplementedError)
    async def not_implemented_handler(request: Request, exc: NotImplementedError):
        return JSONResponse(
            status_code=501,
            content=ErrorResponse(
                detail=str(exc) or "Not implemented",
                error_code="NOT_IMPLEMENTED",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="Internal server error",
                error_code="INTERNAL_ERROR",
            ).model_dump(),
        )
