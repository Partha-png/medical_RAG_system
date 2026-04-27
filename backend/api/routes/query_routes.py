"""
API routes for RAG queries
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.models.conversation_models import QueryRequest, QueryResponse, ConversationHistory
from backend.services.rag_service import RAGService
from backend.services.session_service import SessionService
from backend.services.conversation_service import ConversationService
from backend.services.evaluation_service import EvaluationService
from backend.core.exceptions import SessionNotFound, RetrievalError, LLMError

router = APIRouter(prefix="/api", tags=["Query"])
rag_service = RAGService()
session_service = SessionService()
conversation_service = ConversationService()
evaluation_service = EvaluationService()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a RAG query: retrieve, generate, and auto-evaluate."""
    try:
        session = session_service.get_session(request.session_id)

        # Heavy work goes to a thread so we don't block the event loop.
        answer, chunks, timings = await run_in_threadpool(
            rag_service.query_with_timings,
            request.session_id,
            session.encoder_type,
            request.question,
            request.k,
        )

        # Reference-free quality metrics for this turn.
        metrics = await run_in_threadpool(
            evaluation_service.auto_evaluate,
            request.question,
            answer,
            chunks,
        )
        if isinstance(metrics, dict):
            metrics["latency"] = timings
            metrics["encoder"] = session.encoder_type

        await run_in_threadpool(
            conversation_service.add_message,
            request.session_id,
            "user",
            request.question,
        )
        await run_in_threadpool(
            conversation_service.add_message,
            request.session_id,
            "assistant",
            answer,
            chunks,
            metrics,
        )

        return QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            chunks=chunks,
            metrics=metrics,
        )

    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found",
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """Retrieve chunks only without generating an answer."""
    try:
        session = session_service.get_session(request.session_id)

        chunks = await run_in_threadpool(
            rag_service.retrieve_chunks,
            request.session_id,
            session.encoder_type,
            request.question,
            request.k,
        )

        return {
            "session_id": request.session_id,
            "question": request.question,
            "chunks": chunks,
        }

    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found",
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.get("/sessions/{session_id}/conversation", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    try:
        return await run_in_threadpool(conversation_service.get_conversation, session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}",
        )


@router.delete("/sessions/{session_id}/conversation", status_code=status.HTTP_204_NO_CONTENT)
async def clear_conversation(session_id: str):
    try:
        await run_in_threadpool(conversation_service.clear_conversation, session_id)
        return None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}",
        )
