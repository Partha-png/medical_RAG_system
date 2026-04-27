"""
API routes for document upload and processing
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.services.document_service import DocumentService
from backend.services.session_service import SessionService
from backend.services.rag_service import RAGService
from backend.core.exceptions import DocumentProcessingError, SessionNotFound

router = APIRouter(prefix="/api/documents", tags=["Documents"])
document_service = DocumentService()
session_service = SessionService()


@router.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload and process a document for a session."""
    file_path = None
    try:
        session = session_service.get_session(session_id)

        allowed_extensions = {".pdf", ".txt"}
        original_name = file.filename or ""
        file_ext = "." + original_name.split(".")[-1].lower() if "." in original_name else ""

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not supported. Allowed: {sorted(allowed_extensions)}",
            )

        file_content = await file.read()
        file_path = await run_in_threadpool(
            document_service.save_uploaded_file, file_content, original_name
        )

        result = await run_in_threadpool(
            document_service.process_document,
            file_path,
            session_id,
            session.encoder_type,
            8,
        )

        await run_in_threadpool(
            session_service.update_document_name, session_id, original_name
        )

        # New index data on disk -> drop any stale cached retriever.
        RAGService.invalidate_session_cache(session_id)

        return {
            "message": "Document processed successfully",
            "session_id": session_id,
            "filename": original_name,
            "encoder_type": session.encoder_type,
            "num_embeddings": result.get("num_embeddings", 0),
        }

    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}",
        )
    finally:
        # Don't let temp files pile up.
        if file_path:
            document_service.cleanup_temp_file(file_path)


@router.delete("/{session_id}")
async def delete_document(session_id: str):
    """Delete all documents and FAISS indices for a session."""
    try:
        await run_in_threadpool(document_service.delete_session_data, session_id)
        RAGService.invalidate_session_cache(session_id)
        return {"message": f"Documents deleted for session {session_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}",
        )
