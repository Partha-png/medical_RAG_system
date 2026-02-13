"""
API routes for document upload and processing
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.services.document_service import DocumentService
from backend.services.session_service import SessionService
from backend.core.exceptions import DocumentProcessingError, SessionNotFound

router = APIRouter(prefix="/api/documents", tags=["Documents"])
document_service = DocumentService()
session_service = SessionService()


@router.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and process a document for a session
    """
    try:
        # Verify session exists
        session = session_service.get_session(session_id)
        
        # Validate file type
        allowed_extensions = {".pdf", ".txt"}
        file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file
        file_content = await file.read()
        file_path = document_service.save_uploaded_file(file_content, file.filename)
        
        # Process document
        result = document_service.process_document(
            file_path=file_path,
            session_id=session_id,
            encoder_type=session.encoder_type,
            batch_size=8
        )
        
        # Update session with document name
        session_service.update_document_name(session_id, file.filename)
        
        return {
            "message": "Document processed successfully",
            "session_id": session_id,
            "filename": file.filename,
            "encoder_type": session.encoder_type,
            "num_embeddings": result.get("num_embeddings", 0)
        }
        
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    except DocumentProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )


@router.delete("/{session_id}")
async def delete_document(session_id: str):
    """
    Delete all documents and FAISS indices for a session
    """
    try:
        document_service.delete_session_data(session_id)
        return {"message": f"Documents deleted for session {session_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}"
        )
