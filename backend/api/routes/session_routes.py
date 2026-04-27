"""
API routes for session management
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.models.session_models import SessionCreate, SessionResponse, SessionList
from backend.services.session_service import SessionService
from backend.services.conversation_service import ConversationService
from backend.services.document_service import DocumentService
from backend.services.rag_service import RAGService
from backend.core.exceptions import SessionNotFound

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])
session_service = SessionService()
conversation_service = ConversationService()
document_service = DocumentService()


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(session_create: SessionCreate):
    try:
        return await run_in_threadpool(session_service.create_session, session_create)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get("", response_model=SessionList)
async def list_sessions():
    try:
        return await run_in_threadpool(session_service.list_sessions)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    try:
        return await run_in_threadpool(session_service.get_session, session_id)
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}",
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    try:
        await run_in_threadpool(document_service.delete_session_data, session_id)
        await run_in_threadpool(conversation_service.delete_conversation, session_id)
        await run_in_threadpool(session_service.delete_session, session_id)
        RAGService.invalidate_session_cache(session_id)
        return None
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )
