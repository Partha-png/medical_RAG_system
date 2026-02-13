"""
File-based repository for conversation history
"""
import json
import os
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from backend.core import config
from backend.core.exceptions import SessionNotFound


class ConversationRepository:
    """Handles conversation persistence using JSON files"""
    
    def __init__(self):
        self.conversations_dir = config.CONVERSATIONS_DIR
    
    def add_message(self, session_id: str, role: str, content: str, retrieved_chunks: List[str] = None):
        """Add a message to conversation history"""
        conversation_file = self.conversations_dir / f"{session_id}.json"
        
        # Load existing conversation or create new
        if conversation_file.exists():
            with open(conversation_file, 'r') as f:
                conversation = json.load(f)
        else:
            conversation = {
                "session_id": session_id,
                "messages": []
            }
        
        # Add new message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "retrieved_chunks": retrieved_chunks
        }
        conversation["messages"].append(message)
        
        # Save conversation
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        return message
    
    def get_conversation(self, session_id: str) -> Dict:
        """Get full conversation history for a session"""
        conversation_file = self.conversations_dir / f"{session_id}.json"
        
        if not conversation_file.exists():
            return {
                "session_id": session_id,
                "messages": []
            }
        
        with open(conversation_file, 'r') as f:
            return json.load(f)
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation history for a session"""
        conversation_file = self.conversations_dir / f"{session_id}.json"
        
        if conversation_file.exists():
            conversation_file.unlink()
            return True
        
        return False
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear all messages in a conversation but keep the session"""
        conversation_file = self.conversations_dir / f"{session_id}.json"
        
        conversation = {
            "session_id": session_id,
            "messages": []
        }
        
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f, indent=2)
        
        return True
