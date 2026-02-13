"""
Session management service
"""
import uuid
from typing import List, Optional
from backend.repositories.session_repository import SessionRepository
from backend.models.session_models import SessionCreate, SessionResponse, SessionList
from backend.core.exceptions import SessionNotFound
from datetime import datetime


class SessionService:
    """Business logic for session management"""
    
    def __init__(self):
        self.repository = SessionRepository()
    
    def create_session(self, session_create: SessionCreate, document_name: Optional[str] = None) -> SessionResponse:
        """Create a new session with a unique ID"""
        session_id = str(uuid.uuid4())
        
        session_data = self.repository.create_session(
            session_id=session_id,
            encoder_type=session_create.encoder_type,
            document_name=document_name
        )
        
        return SessionResponse(
            session_id=session_data["session_id"],
            encoder_type=session_data["encoder_type"],
            document_name=session_data.get("document_name"),
            created_at=datetime.fromisoformat(session_data["created_at"])
        )
    
    def get_session(self, session_id: str) -> SessionResponse:
        """Get session by ID"""
        session_data = self.repository.get_session(session_id)
        
        return SessionResponse(
            session_id=session_data["session_id"],
            encoder_type=session_data["encoder_type"],
            document_name=session_data.get("document_name"),
            created_at=datetime.fromisoformat(session_data["created_at"])
        )
    
    def list_sessions(self) -> SessionList:
        """List all sessions"""
        sessions_data = self.repository.list_sessions()
        
        sessions = [
            SessionResponse(
                session_id=s["session_id"],
                encoder_type=s["encoder_type"],
                document_name=s.get("document_name"),
                created_at=datetime.fromisoformat(s["created_at"])
            )
            for s in sessions_data
        ]
        
        return SessionList(sessions=sessions, total=len(sessions))
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.repository.delete_session(session_id)
    
    def update_document_name(self, session_id: str, document_name: str) -> SessionResponse:
        """Update the document name for a session"""
        session_data = self.repository.update_session(session_id, document_name=document_name)
        
        return SessionResponse(
            session_id=session_data["session_id"],
            encoder_type=session_data["encoder_type"],
            document_name=session_data.get("document_name"),
            created_at=datetime.fromisoformat(session_data["created_at"])
        )
