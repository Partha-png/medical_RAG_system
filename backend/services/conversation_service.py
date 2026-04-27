"""
Conversation management service
"""
from typing import Any, Dict, List, Optional
from backend.repositories.conversation_repository import ConversationRepository
from backend.models.conversation_models import Message, ConversationHistory
from datetime import datetime


class ConversationService:
    """Business logic for conversation management"""

    def __init__(self):
        self.repository = ConversationRepository()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieved_chunks: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Message:
        message_data = self.repository.add_message(
            session_id=session_id,
            role=role,
            content=content,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
        )

        return Message(
            role=message_data["role"],
            content=message_data["content"],
            timestamp=datetime.fromisoformat(message_data["timestamp"]),
            retrieved_chunks=message_data.get("retrieved_chunks"),
            metrics=message_data.get("metrics"),
        )

    def get_conversation(self, session_id: str) -> ConversationHistory:
        conversation_data = self.repository.get_conversation(session_id)

        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                retrieved_chunks=m.get("retrieved_chunks"),
                metrics=m.get("metrics"),
            )
            for m in conversation_data.get("messages", [])
        ]

        return ConversationHistory(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
        )

    def delete_conversation(self, session_id: str) -> bool:
        return self.repository.delete_conversation(session_id)

    def clear_conversation(self, session_id: str) -> bool:
        return self.repository.clear_conversation(session_id)
