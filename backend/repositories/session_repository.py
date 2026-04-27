"""
File-based repository for session management
"""
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, List
from backend.core import config
from backend.core.exceptions import SessionNotFound


def _atomic_write_json(path, data) -> None:
    """Write JSON atomically: write to a temp file then rename into place."""
    path = str(path)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class SessionRepository:
    """Handles session persistence using JSON files"""

    def __init__(self):
        self.sessions_dir = config.SESSIONS_DIR

    def create_session(self, session_id: str, encoder_type: str, document_name: Optional[str] = None) -> dict:
        session_data = {
            "session_id": session_id,
            "encoder_type": encoder_type,
            "document_name": document_name,
            "created_at": datetime.now().isoformat(),
        }

        session_file = self.sessions_dir / f"{session_id}.json"
        _atomic_write_json(session_file, session_data)
        return session_data

    def get_session(self, session_id: str) -> dict:
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            raise SessionNotFound(f"Session {session_id} not found")

        with open(session_file, "r") as f:
            return json.load(f)

    def list_sessions(self) -> List[dict]:
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    sessions.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                # Skip corrupt files instead of crashing the whole listing.
                continue

        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            raise SessionNotFound(f"Session {session_id} not found")

        session_file.unlink()
        return True

    def update_session(self, session_id: str, **kwargs) -> dict:
        session_data = self.get_session(session_id)
        session_data.update(kwargs)

        session_file = self.sessions_dir / f"{session_id}.json"
        _atomic_write_json(session_file, session_data)
        return session_data
