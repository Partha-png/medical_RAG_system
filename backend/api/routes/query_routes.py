"""
API routes for RAG queries
"""
from fastapi import APIRouter, HTTPException, status
from backend.models.conversation_models import QueryRequest, QueryResponse, ConversationHistory
from backend.services.rag_service import RAGService
from backend.services.session_service import SessionService
from backend.services.conversation_service import ConversationService
from backend.core.exceptions import SessionNotFound, RetrievalError, LLMError

router = APIRouter(prefix="/api", tags=["Query"])
rag_service = RAGService()
session_service = SessionService()
conversation_service = ConversationService()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a RAG query: retrieve chunks and generate answer
    """
    try:
        # Verify session exists
        session = session_service.get_session(request.session_id)
        
        # Perform RAG query
        answer, chunks = rag_service.query(
            session_id=request.session_id,
            encoder_type=session.encoder_type,
            question=request.question,
            k=request.k
        )
        
        # Save user message
        conversation_service.add_message(
            session_id=request.session_id,
            role="user",
            content=request.question
        )
        
        # Save assistant response
        conversation_service.add_message(
            session_id=request.session_id,
            role="assistant",
            content=answer,
            retrieved_chunks=chunks
        )
        
        return QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            chunks=chunks
        )
        
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """
    Retrieve chunks only without generating an answer
    """
    try:
        # Verify session exists
        session = session_service.get_session(request.session_id)
        
        # Retrieve chunks
        chunks = rag_service.retrieve_chunks(
            session_id=request.session_id,
            encoder_type=session.encoder_type,
            query=request.question,
            k=request.k
        )
        
        return {
            "session_id": request.session_id,
            "question": request.question,
            "chunks": chunks
        }
        
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.get("/sessions/{session_id}/conversation", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """
    Get conversation history for a session
    """
    try:
        return conversation_service.get_conversation(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.delete("/sessions/{session_id}/conversation", status_code=status.HTTP_204_NO_CONTENT)
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session
    """
    try:
        conversation_service.clear_conversation(session_id)
        return None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}"
        )
