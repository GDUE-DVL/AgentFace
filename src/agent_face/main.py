"""
AgentFace — FastAPI application entry point.

Launches the FastAPI server with the LangGraph brain and MAF body
initialized during the lifespan.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent_face.config import settings
from agent_face.api.router import router as api_router
from agent_face.api.middleware import setup_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and teardown resources."""
    # Startup
    from agent_face.langgraph_brain.graph import build_graph
    from agent_face.maf_body.orchestrator import MAFOrchestrator
    from agent_face.bridge.maf_client import MAFBridgeClient

    # Initialize MAF orchestrator
    orchestrator = MAFOrchestrator()
    await orchestrator.start()

    # Initialize bridge
    bridge = MAFBridgeClient(orchestrator=orchestrator)

    # Build the LangGraph with SQLite persistence (async versions)
    import os
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    os.makedirs("data", exist_ok=True)
    db_path = "data/agent_face.db"

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(db_path)
    store_cm = AsyncSqliteStore.from_conn_string(db_path)

    checkpointer = await checkpointer_cm.__aenter__()
    store = await store_cm.__aenter__()

    graph = build_graph(checkpointer=checkpointer, store=store)

    # Store for cleanup
    app.state.checkpointer_cm = checkpointer_cm
    app.state.store_cm = store_cm

    # Store in app state for dependency injection
    app.state.graph = graph
    app.state.bridge = bridge
    app.state.orchestrator = orchestrator

    yield

    # Shutdown — close async SQLite context managers
    if hasattr(app.state, 'checkpointer_cm'):
        await app.state.checkpointer_cm.__aexit__(None, None, None)
    if hasattr(app.state, 'store_cm'):
        await app.state.store_cm.__aexit__(None, None, None)
    await orchestrator.stop()


app = FastAPI(
    title="AgentFace",
    description="Beautification Agent System — LangGraph + Microsoft Agent Framework",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
setup_exception_handlers(app)

# Routes
app.include_router(api_router, prefix="/api/v1")

# Static files & Web UI
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/app")
@app.get("/app/")
async def web_ui():
    """Serve the Web UI."""
    from fastapi.responses import FileResponse
    return FileResponse(static_dir / "index.html")


@app.get("/")
async def root():
    """Root endpoint — redirects to API docs."""
    return {
        "service": "AgentFace",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_face.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=True,
    )
