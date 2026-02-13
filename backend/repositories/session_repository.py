"""
File-based repository for session management
"""
import json
import os
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from backend.core import config
from backend.core.exceptions import SessionNotFound


class SessionRepository:
    """Handles session persistence using JSON files"""
    
    def __init__(self):
        self.sessions_dir = config.SESSIONS_DIR
        
    def create_session(self, session_id: str, encoder_type: str, document_name: Optional[str] = None) -> dict:
        """Create a new session"""
        session_data = {
            "session_id": session_id,
            "encoder_type": encoder_type,
            "document_name": document_name,
            "created_at": datetime.now().isoformat()
        }
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_data
    
    def get_session(self, session_id: str) -> dict:
        """Get session by ID"""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise SessionNotFound(f"Session {session_id} not found")
        
        with open(session_file, 'r') as f:
            return json.load(f)
    
    def list_sessions(self) -> List[dict]:
        """List all sessions"""
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            with open(session_file, 'r') as f:
                sessions.append(json.load(f))
        
        # Sort by created_at descending
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            raise SessionNotFound(f"Session {session_id} not found")
        
        session_file.unlink()
        return True
    
    def update_session(self, session_id: str, **kwargs) -> dict:
        """Update session metadata"""
        session_data = self.get_session(session_id)
        session_data.update(kwargs)
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_data
