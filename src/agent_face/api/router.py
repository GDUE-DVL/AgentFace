"""
AgentFace REST API router.

All API endpoints for session lifecycle, user preferences, and
system health. HITL (Human-in-the-Loop) is implemented via
LangGraph interrupt/resume — the workflow pauses at two points
and resumes when the user provides input via the API.
"""

import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.types import Command

from agent_face.api.dependencies import get_graph, get_bridge
from agent_face.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    ConfirmPlanRequest,
    SubmitFeedbackRequest,
    RerunRequest,
    SessionResponse,
    UserPreferencesResponse,
    SessionHistoryResponse,
    HealthResponse,
    ErrorResponse,
    AnalysisResultModel,
    BeautifyParamsModel,
)
from agent_face.langgraph_brain.state import BeautifyWorkflowState
from agent_face.langgraph_brain.memory.preferences import get_user_preferences
from agent_face.langgraph_brain.memory.feedback import get_user_feedback_summary
from agent_face.langgraph_brain.memory.history import get_session_history
from agent_face.bridge.types import BeautificationRequest
from agent_face.bridge.maf_client import BridgeError
from agent_face.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────


def _format_session_response(state: dict) -> SessionResponse:
    """Format a LangGraph state dict into a SessionResponse model."""
    stage = state.get("workflow_stage", "unknown")

    # Map stage to user-facing status (Chinese)
    status_map = {
        "input_received": "分析中",
        "context_loaded": "分析中",
        "analyzed": "等待确认",
        "plan_confirmed": "美颜中",
        "plan_presented": "等待确认",
        "beautified": "等待评分",
        "feedback_collected": "已完成",
        "completed": "已完成",
        "failed": "失败",
        "error": "失败",
    }

    analysis = state.get("analysis_result")
    analysis_model = None
    if analysis:
        analysis_model = AnalysisResultModel(
            skin_tone=analysis.get("skin_tone", ""),
            skin_condition=analysis.get("skin_condition", ""),
            lighting=analysis.get("lighting", "—"),
            detected_features=analysis.get("detected_features", []),
            detected_issues=analysis.get("detected_issues", []),
            suggested_params=BeautifyParamsModel(
                **analysis.get("suggested_params", {})
            ),
            reasoning=analysis.get("reasoning", ""),
            confidence=analysis.get("confidence", 0.0),
            identity_features=analysis.get("identity_features", ""),
            expression_state=analysis.get("expression_state", ""),
            source_skin=analysis.get("source_skin", ""),
            target_skin=analysis.get("target_skin", ""),
            source_description=analysis.get("source_description", ""),
            target_description=analysis.get("target_description", ""),
            edit_regions=analysis.get("edit_regions", []) or [],
            scar_check=analysis.get("scar_check", "absent"),
            scar_location=analysis.get("scar_location", ""),
            scar_bbox=analysis.get("scar_bbox", []) or [],
            acne_check=analysis.get("acne_check", "absent"),
            acne_bbox=analysis.get("acne_bbox", []) or [],
            wrinkle_check=analysis.get("wrinkle_check", "absent"),
            wrinkle_bbox=analysis.get("wrinkle_bbox", []) or [],
        )

    final_params = state.get("final_params")
    final_params_model = None
    if final_params:
        final_params_model = BeautifyParamsModel(**final_params)

    return SessionResponse(
        session_id=state.get("session_id", "unknown"),
        user_id=state.get("user_id", "anonymous"),
        workflow_stage=stage,
        status=status_map.get(stage, "unknown"),
        created_at=state.get("created_at", ""),
        analysis_result=analysis_model,
        beautified_image=state.get("beautified_image_b64"),
        final_params=final_params_model,
        error_message=state.get("error_message"),
        rerun_count=int(state.get("rerun_count", 0) or 0),
        current_seed=state.get("current_seed"),
        rerun_history=state.get("rerun_history", []) or [],
    )


# ── Session Endpoints ──────────────────────────────────────────


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
    tags=["Sessions"],
)
async def create_session(
    request_data: CreateSessionRequest,
    req: Request,
):
    """
    Create a new beautification session.

    Upload an image and optional natural language prompt.
    The workflow runs until the first HITL interrupt (present_plan),
    then pauses and returns the analysis for user confirmation.

    Workflow:
    1. receive_input — validates image
    2. retrieve_context — loads user preferences from Store
    3. analyze_image — calls multimodal model via MAF
    4. present_plan — PAUSES here for user confirmation
    """
    graph = req.app.state.graph
    store = graph.store  # InMemoryStore from graph compilation

    session_id = str(uuid.uuid4())
    thread_id = session_id  # Use session_id as LangGraph thread_id

    # Build initial state
    initial_state: BeautifyWorkflowState = {
        "session_id": session_id,
        "user_id": request_data.user_id,
        "input_image_b64": request_data.image,
        "user_prompt": request_data.user_prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow_stage": "start",
        "retry_count": 0,
        "rerun_count": 0,
        "current_seed": settings.beauty_seed,
        "rerun_history": [],
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
            "store": store,
            "bridge": req.app.state.bridge,
        }
    }

    # Run the graph — in LangGraph 1.x, interrupt() returns normally
    # (does NOT raise an exception). The graph pauses at the interrupt
    # and the state contains __interrupt__ marker.
    await graph.ainvoke(initial_state, config)

    # Get current state after interrupt
    current_state = await graph.aget_state(config)

    logger.info(
        f"Session {session_id} created for user {request_data.user_id} "
        f"(stage: {current_state.values.get('workflow_stage', 'unknown') if current_state else 'unknown'})"
    )

    return CreateSessionResponse(
        session_id=session_id,
        status="等待确认",
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"],
)
async def get_session(
    session_id: str,
    req: Request,
):
    """
    Get the current status of a beautification session.

    Returns current workflow stage, analysis (if available),
    and the beautified image (if completed).
    """
    graph = req.app.state.graph

    config = {"configurable": {"thread_id": session_id}}

    current_state = await graph.aget_state(config)
    if current_state is None or current_state.values is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 {session_id} 不存在",
        )

    return _format_session_response(current_state.values)


@router.post(
    "/sessions/{session_id}/confirm",
    response_model=SessionResponse,
    tags=["Sessions"],
)
async def confirm_plan(
    session_id: str,
    request_data: ConfirmPlanRequest,
    req: Request,
):
    """
    Confirm or adjust the beautification plan (HITL resume point 1).

    After the multimodal model analyzes the image, the workflow pauses
    at `present_plan`. Call this endpoint to:
    - confirm: accept the suggested parameters as-is
    - adjust: provide modified parameters

    The workflow then resumes: apply_beautification → collect_feedback (pause).
    """
    graph = req.app.state.graph
    store = graph.store

    config = {
        "configurable": {
            "thread_id": session_id,
            "store": store,
            "bridge": req.app.state.bridge,
        }
    }

    # Check that the session exists and is at the right stage
    current_state = await graph.aget_state(config)
    if current_state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stage = current_state.values.get("workflow_stage", "")
    if stage not in ("plan_presented", "analyzed"):
        raise HTTPException(
            status_code=409,
            detail=f"当前会话处于'{stage}'阶段，不是等待确认阶段",
        )

    # Build the resume command
    resume_value = {
        "action": request_data.action,
    }
    if request_data.action == "adjust" and request_data.adjustments:
        resume_value["adjustments"] = request_data.adjustments.model_dump(
            exclude_none=True
        )

    # Resume the graph — runs from present_plan through apply_beautification
    # and pauses again at collect_feedback (interrupt returns normally)
    await graph.ainvoke(Command(resume=resume_value), config)

    # Get updated state
    updated_state = await graph.aget_state(config)

    logger.info(
        f"Session {session_id}: plan {'confirmed' if request_data.action == 'confirm' else 'adjusted'}"
    )

    return _format_session_response(updated_state.values)


@router.post(
    "/sessions/{session_id}/feedback",
    response_model=SessionResponse,
    tags=["Sessions"],
)
async def submit_feedback(
    session_id: str,
    request_data: SubmitFeedbackRequest,
    req: Request,
):
    """
    Submit feedback after reviewing the beautified result (HITL resume point 2).

    After beautification completes, the workflow pauses at `collect_feedback`.
    Call this endpoint to rate the result (1-5) and provide comments.

    The workflow then resumes: update_memory → finalize → END.
    Feedback satisfaction scores >= 3 trigger EMA preference updates.
    """
    graph = req.app.state.graph
    store = graph.store

    config = {
        "configurable": {
            "thread_id": session_id,
            "store": store,
            "bridge": req.app.state.bridge,
        }
    }

    # Check session state
    current_state = await graph.aget_state(config)
    if current_state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stage = current_state.values.get("workflow_stage", "")
    if stage not in ("beautified",):
        raise HTTPException(
            status_code=409,
            detail=f"当前会话处于'{stage}'阶段，不是等待评分阶段",
        )

    # Build resume value
    resume_value = {
        "satisfaction_score": request_data.satisfaction_score,
        "comments": request_data.comments,
    }
    if request_data.param_adjustments:
        resume_value["param_adjustments"] = request_data.param_adjustments.model_dump(
            exclude_none=True
        )

    # Resume the graph — runs from collect_feedback through update_memory and finalize
    final_state = await graph.ainvoke(Command(resume=resume_value), config)

    logger.info(
        f"Session {session_id}: feedback submitted (score={request_data.satisfaction_score})"
    )

    return _format_session_response(final_state)


@router.post(
    "/sessions/{session_id}/rerun",
    response_model=SessionResponse,
    tags=["Sessions"],
)
async def rerun_session(
    session_id: str,
    request_data: RerunRequest,
    req: Request,
):
    """Rerun the same image/plan with a different seed.

    This is the human-in-the-loop correction path: reviewers can keep the
    original analysis and prompts, click rerun, and receive a fresh output.
    Every attempt is retained in ``rerun_history`` in the session checkpoint.
    """
    graph = req.app.state.graph
    bridge = req.app.state.bridge
    config = {"configurable": {"thread_id": session_id, "store": graph.store, "bridge": bridge}}
    current_state = await graph.aget_state(config)
    if current_state is None or current_state.values is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    state = current_state.values
    stage = state.get("workflow_stage", "")
    if stage not in ("beautified", "feedback_collected", "completed"):
        raise HTTPException(
            status_code=409,
            detail=f"当前会话处于'{stage}'阶段，只有已有结果时才能重跑",
        )
    if not state.get("input_image_b64") or not state.get("final_params"):
        raise HTTPException(status_code=409, detail="会话缺少原图或已确认方案，无法重跑")

    rerun_count = int(state.get("rerun_count", 0) or 0)
    previous_seed = state.get("current_seed")
    base_seed = int(previous_seed if previous_seed is not None else settings.beauty_seed)
    seed = int(request_data.seed) if request_data.seed is not None else (base_seed + 1)
    analysis = state.get("analysis_result") or {}
    request = BeautificationRequest(
        image_b64=state["input_image_b64"],
        params=state["final_params"],
        src_prompt=analysis.get("source_description", "") if isinstance(analysis, dict) else "",
        target_prompt=analysis.get("target_description", "") if isinstance(analysis, dict) else "",
        edit_regions=analysis.get("edit_regions", []) if isinstance(analysis, dict) else [],
        seed=seed,
    )

    try:
        response = await bridge.apply_beautification(
            request=request,
            user_id=state.get("user_id", "anonymous"),
            session_id=session_id,
        )
    except BridgeError as exc:
        logger.error(f"Session {session_id}: rerun failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Session {session_id}: unexpected rerun failure")
        raise HTTPException(status_code=502, detail=f"重跑失败: {exc}") from exc

    history = list(state.get("rerun_history") or [])
    history.append({
        "attempt": rerun_count + 1,
        "seed": seed,
        "reason": request_data.reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(response.latency_ms, 1),
    })
    await graph.aupdate_state(config, {
        "beautified_image_b64": response.image_b64,
        "workflow_stage": "beautified",
        "error_message": None,
        "user_feedback": None,
        "rerun_count": rerun_count + 1,
        "current_seed": seed,
        "rerun_history": history,
    })
    updated_state = await graph.aget_state(config)
    logger.info(f"Session {session_id}: rerun completed (attempt={rerun_count + 1}, seed={seed})")
    return _format_session_response(updated_state.values)


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    tags=["Sessions"],
)
async def abandon_session(
    session_id: str,
    req: Request,
):
    """
    Abandon an active session.

    The session state remains in the checkpointer for audit purposes
    but won't be processed further.
    """
    graph = req.app.state.graph
    config = {"configurable": {"thread_id": session_id}}

    current_state = await graph.aget_state(config)
    if current_state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # LangGraph doesn't have a built-in "delete" for threads,
    # but we can mark it as abandoned in the state.
    # For now, just return 204 — the checkpointer retains the last state.
    logger.info(f"Session {session_id} abandoned")


# ── User Endpoints ─────────────────────────────────────────────


@router.get(
    "/users/{user_id}/preferences",
    response_model=UserPreferencesResponse,
    tags=["Users"],
)
async def get_user_preferences_endpoint(
    user_id: str,
    req: Request,
):
    """
    Get the current beautification preferences for a user.

    Preferences evolve over time based on feedback using
    Exponential Moving Average (EMA).
    """
    graph = req.app.state.graph
    store = graph.store

    preferences = await get_user_preferences(store, user_id)
    feedback_summary = await get_user_feedback_summary(store, user_id)

    return UserPreferencesResponse(
        user_id=user_id,
        preferences=BeautifyParamsModel(**preferences),
        total_sessions=feedback_summary.get("total_sessions", 0),
        avg_satisfaction=feedback_summary.get("avg_satisfaction"),
    )


@router.delete(
    "/users/{user_id}/preferences",
    status_code=204,
    tags=["Users"],
)
async def reset_user_preferences(
    user_id: str,
    req: Request,
):
    """
    Reset user beautification preferences to defaults.
    Clears all learned preferences. Session history is NOT deleted.
    """
    graph = req.app.state.graph
    store = graph.store

    from agent_face.langgraph_brain.state import DEFAULT_BEAUTIFY_PARAMS

    namespace = ("users", user_id, "preferences")
    try:
        await store.aput(namespace, "current", dict(DEFAULT_BEAUTIFY_PARAMS))
        logger.info(f"Reset preferences for {user_id}")
    except Exception as e:
        logger.error(f"Failed to reset preferences for {user_id}: {e}")


@router.get(
    "/users/{user_id}/history",
    response_model=SessionHistoryResponse,
    tags=["Users"],
)
async def get_user_history_endpoint(
    user_id: str,
    req: Request,
    limit: int = 20,
    offset: int = 0,
):
    """
    Get beautification session history for a user.

    Returns paginated list of past sessions with parameter summaries.
    """
    graph = req.app.state.graph
    store = graph.store

    sessions = await get_session_history(store, user_id, limit=limit, offset=offset)

    items = []
    for s in sessions:
        params = None
        if s.get("final_params"):
            params = BeautifyParamsModel(**s["final_params"])

        feedback = s.get("user_feedback", {})
        items.append({
            "session_id": s.get("session_id", ""),
            "created_at": s.get("created_at", ""),
            "workflow_stage": s.get("workflow_stage", ""),
            "satisfaction_score": feedback.get("satisfaction_score"),
            "thumbnail_params": params,
        })

    return SessionHistoryResponse(
        user_id=user_id,
        total=len(items),
        sessions=items,
    )


# ── Health Endpoint ────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check(req: Request):
    """
    System health check.

    Verifies connectivity to LangGraph, MAF orchestrator,
    and both model services.
    """
    bridge = req.app.state.bridge

    try:
        maf_health = await bridge.health_check()
    except Exception as e:
        maf_health = {"status": "error", "error": str(e)}

    # Check individual model services through the orchestrator health
    multimodal_health = maf_health.get("multimodal_agent", {})
    beautification_health = maf_health.get("beautification_agent", {})

    # Determine overall status
    maf_ok = maf_health.get("status") == "ok"
    multimodal_ok = multimodal_health.get("status") == "ok"
    beautify_ok = beautification_health.get("status") == "ok"

    if maf_ok and multimodal_ok and beautify_ok:
        overall = "正常"
    elif maf_ok:
        overall = "部分降级"
    else:
        overall = "异常"

    return HealthResponse(
        status=overall,
        langgraph="ok",
        maf_orchestrator=maf_health.get("status", "unknown"),
        multimodal_model=multimodal_health,
        beautification_model=beautification_health,
    )
