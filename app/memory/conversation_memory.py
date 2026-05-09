"""
Conversation Memory - Maintains multi-turn conversation state

Handles:
- Short-term conversation history
- Long-term user preferences
- Session management
- Context window optimization
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid


class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = []
    created_at: datetime
    last_activity: datetime
    context_summary: str = ""
    metadata: Dict[str, Any] = {}


class ConversationMemory:
    """
    Manages conversation state across multiple turns.
    
    Features:
    - Sliding window history
    - Summary-based compression
    - User preference tracking
    - Session persistence
    """
    
    def __init__(
        self,
        max_messages: int = 50,
        summary_threshold: int = 40,
        ttl_hours: int = 24
    ):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.ttl_seconds = ttl_hours * 3600
        self.sessions: Dict[str, ConversationSession] = {}
    
    def create_session(
        self, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new conversation session."""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata=metadata or {}
        )
        
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to conversation."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.last_activity = datetime.now()
        
        # Compress if needed
        if len(session.messages) > self.summary_threshold:
            self._compress_history(session)
        
        # Enforce max messages
        if len(session.messages) > self.max_messages:
            session.messages = session.messages[-self.max_messages:]
        
        return message
    
    def get_history(
        self, 
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation history."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        if limit:
            return session.messages[-limit:]
        
        return session.messages
    
    def get_context_for_turn(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> List[Dict[str, str]]:
        """
        Get formatted context for next LLM turn.
        
        Returns list of {"role": ..., "content": ...} dicts
        optimized for token budget.
        """
        history = self.get_history(session_id)
        
        if not history:
            return []
        
        # Convert to dict format
        messages = [{"role": m.role, "content": m.content} for m in history]
        
        # Simple token-based truncation (reverse to keep recent)
        total_chars = sum(len(m["content"]) for m in messages)
        avg_tokens = total_chars // 4
        
        if avg_tokens <= max_tokens:
            return messages
        
        # Truncate from beginning
        truncated = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = len(msg["content"]) // 4
            if current_tokens + msg_tokens <= max_tokens:
                truncated.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return truncated
    
    def _compress_history(self, session: ConversationSession) -> None:
        """Compress old messages into summary."""
        if len(session.messages) < 10:
            return
        
        # Keep last N messages, summarize the rest
        keep_count = 10
        old_messages = session.messages[:-keep_count]
        
        # Generate simple summary (in production, use LLM)
        summary_parts = []
        for msg in old_messages[:5]:  # Sample first 5 old messages
            if msg.role == "user":
                summary_parts.append(f"User asked about: {msg.content[:50]}...")
        
        session.context_summary = "\n".join(summary_parts)
        
        # Replace old messages with summary message
        summary_message = Message(
            role="system",
            content=f"[Conversation Summary]\n{session.context_summary}",
            timestamp=datetime.now(),
            metadata={"type": "summary"}
        )
        
        session.messages = [summary_message] + session.messages[-keep_count:]
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close and archive session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            age_seconds = (now - session.last_activity).total_seconds()
            if age_seconds > self.ttl_seconds:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / len(self.sessions) if self.sessions else 0,
            "max_messages_limit": self.max_messages,
        }
