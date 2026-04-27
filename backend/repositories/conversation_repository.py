"""
File-based repository for conversation history
"""
import json
import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.core import config


_write_lock = threading.Lock()


def _atomic_write_json(path, data) -> None:
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


class ConversationRepository:
    """Handles conversation persistence using JSON files"""

    def __init__(self):
        self.conversations_dir = config.CONVERSATIONS_DIR

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieved_chunks: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        conversation_file = self.conversations_dir / f"{session_id}.json"

        with _write_lock:
            if conversation_file.exists():
                with open(conversation_file, "r") as f:
                    try:
                        conversation = json.load(f)
                    except json.JSONDecodeError:
                        conversation = {"session_id": session_id, "messages": []}
            else:
                conversation = {"session_id": session_id, "messages": []}

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "retrieved_chunks": retrieved_chunks,
                "metrics": metrics,
            }
            conversation["messages"].append(message)

            _atomic_write_json(conversation_file, conversation)

        return message

    def get_conversation(self, session_id: str) -> Dict:
        conversation_file = self.conversations_dir / f"{session_id}.json"

        if not conversation_file.exists():
            return {"session_id": session_id, "messages": []}

        with open(conversation_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"session_id": session_id, "messages": []}

    def delete_conversation(self, session_id: str) -> bool:
        conversation_file = self.conversations_dir / f"{session_id}.json"
        if conversation_file.exists():
            conversation_file.unlink()
            return True
        return False

    def clear_conversation(self, session_id: str) -> bool:
        conversation_file = self.conversations_dir / f"{session_id}.json"
        conversation = {"session_id": session_id, "messages": []}
        _atomic_write_json(conversation_file, conversation)
        return True
