"""
Phase 3: Conversation Memory Service

Manages long-term conversation history and user context.
Stores interactions in vector format for semantic retrieval.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.components.embeddings import get_embedding_model


class Message(BaseModel):
    """Single message in conversation."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class ConversationSession(BaseModel):
    """Complete conversation session."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = []
    created_at: float
    updated_at: float
    topic: Optional[str] = None
    summary: Optional[str] = None


class ConversationStore:
    """
    Stores and retrieves conversation history.
    
    Features:
    - Session management
    - Message history with metadata
    - Topic extraction (placeholder for LLM)
    - Automatic summarization (placeholder)
    """
    
    def __init__(self, max_messages_per_session: int = 100):
        self.max_messages = max_messages_per_session
        self._sessions: Dict[str, ConversationSession] = {}
        self._embedding_model = get_embedding_model()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        now = time.time()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now
        )
        
        self._sessions[session_id] = session
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to an existing session."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._sessions[session_id]
        
        # Enforce max messages (sliding window)
        if len(session.messages) >= self.max_messages:
            session.messages.pop(0)
        
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        
        session.messages.append(message)
        session.updated_at = time.time()
        
        # Auto-update topic from first user message
        if role == "user" and len(session.messages) == 1:
            session.topic = self._extract_topic(content)
        
        return message
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session."""
        return self._sessions.get(session_id)
    
    def get_messages(self, session_id: str, 
                     limit: Optional[int] = None) -> List[Message]:
        """Get messages from a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_context(self, session_id: str, 
                    last_n: int = 5) -> List[Dict[str, str]]:
        """Get conversation context in LLM-friendly format."""
        messages = self.get_messages(session_id, limit=last_n)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def summarize_session(self, session_id: str) -> Optional[str]:
        """
        Generate summary of conversation.
        
        Note: In Phase 3, this is a placeholder. 
        Phase 4 will integrate LLM-based summarization.
        """
        session = self._sessions.get(session_id)
        if not session or not session.messages:
            return None
        
        # Simple extractive summary (first and last messages)
        if len(session.messages) >= 2:
            first = session.messages[0].content[:100]
            last = session.messages[-1].content[:100]
            session.summary = f"Discussion: {first}... → {last}..."
        else:
            session.summary = session.messages[0].content[:200]
        
        return session.summary
    
    def _extract_topic(self, text: str) -> str:
        """Extract topic from text (simple keyword extraction)."""
        # Remove common words and extract key terms
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'when'}
        words = text.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        if key_words:
            return " ".join(key_words[:5])
        return "general"
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self, user_id: Optional[str] = None,
                      limit: int = 10) -> List[ConversationSession]:
        """List sessions, optionally filtered by user."""
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        return sessions[:limit]
    
    def search_messages(self, session_id: str, query: str,
                        limit: int = 5) -> List[Message]:
        """
        Search messages within a session by semantic similarity.
        
        Note: Full semantic search requires embeddings (Phase 3+).
        This is a keyword-based fallback for Phase 3.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for msg in session.messages:
            if query_lower in msg.content.lower():
                matches.append(msg)
        
        return matches[-limit:]
    
    def clear_all(self):
        """Clear all sessions (for testing)."""
        self._sessions.clear()


# Global store instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create global conversation store instance."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
