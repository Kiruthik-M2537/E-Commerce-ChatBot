"""Conversational Memory System for AI Shopping Assistant.

Manages message history, turn windowing, active entities (budget, category, products, orders),
and integrates with Streamlit session_state while remaining fully usable in headless unit tests.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    products: Optional[List[Dict[str, Any]]] = None
    order: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "products": self.products,
            "order": self.order,
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Session-level conversation memory with entity tracking and turn windowing."""

    def __init__(self, max_turns: int = 10, session_state: Optional[Any] = None):
        """Initialize memory.

        If session_state is provided (e.g. st.session_state), it synchronizes with it.
        """
        self.max_turns = max_turns
        self._session_state = session_state
        self._local_messages: List[ChatMessage] = []
        self._local_context: Dict[str, Any] = {
            "active_budget": None,
            "active_category": None,
            "last_products": [],
            "active_order_id": None,
        }

        # If connected to session_state, initialize or sync
        if self._session_state is not None:
            if "chat_messages" not in self._session_state:
                self._session_state.chat_messages = []
            if "chat_context" not in self._session_state:
                self._session_state.chat_context = {
                    "active_budget": None,
                    "active_category": None,
                    "last_products": [],
                    "active_order_id": None,
                }

    @property
    def messages(self) -> List[ChatMessage]:
        if self._session_state is not None and hasattr(self._session_state, "chat_messages"):
            return self._session_state.chat_messages
        return self._local_messages

    @property
    def context(self) -> Dict[str, Any]:
        if self._session_state is not None and hasattr(self._session_state, "chat_context"):
            return self._session_state.chat_context
        return self._local_context

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Record a user query."""
        msg = ChatMessage(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        self._append_message(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        products: Optional[List[Dict[str, Any]]] = None,
        order: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Record an assistant response with optional associated products or order."""
        msg = ChatMessage(
            role="assistant",
            content=content,
            products=products,
            order=order,
            metadata=metadata or {},
        )
        self._append_message(msg)

        # Update entity context when products or orders are discussed
        if products:
            self.set_context_entity("last_products", products)
            if len(products) > 0 and "category" in products[0]:
                self.set_context_entity("active_category", products[0]["category"])
        if order and "order_id" in order:
            self.set_context_entity("active_order_id", order["order_id"])

        return msg

    def _append_message(self, message: ChatMessage) -> None:
        if self._session_state is not None and hasattr(self._session_state, "chat_messages"):
            self._session_state.chat_messages.append(message)
            # Windowing: Keep up to 2 * max_turns messages
            max_messages = self.max_turns * 2
            if len(self._session_state.chat_messages) > max_messages:
                self._session_state.chat_messages = self._session_state.chat_messages[-max_messages:]
        else:
            self._local_messages.append(message)
            max_messages = self.max_turns * 2
            if len(self._local_messages) > max_messages:
                self._local_messages = self._local_messages[-max_messages:]

    def get_messages(self, limit: Optional[int] = None) -> List[ChatMessage]:
        """Return chat messages, optionally limited to the most recent N items."""
        msgs = self.messages
        if limit and limit > 0:
            return msgs[-limit:]
        return list(msgs)

    def get_conversation_history_for_prompt(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Format recent history as a list of role/content dicts for LLM prompting."""
        turns = max_turns or self.max_turns
        max_messages = turns * 2
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]

    def set_context_entity(self, key: str, value: Any) -> None:
        """Update context entity (e.g. active_budget, active_category, last_products)."""
        if self._session_state is not None and hasattr(self._session_state, "chat_context"):
            self._session_state.chat_context[key] = value
        else:
            self._local_context[key] = value

    def get_context_entity(self, key: str, default: Any = None) -> Any:
        """Retrieve a context entity."""
        return self.context.get(key, default)

    def get_last_recommended_products(self) -> List[Dict[str, Any]]:
        """Retrieve products from the most recent recommendation turn."""
        return self.get_context_entity("last_products", []) or []

    def get_active_budget(self) -> Optional[float]:
        """Retrieve active user budget constraint."""
        return self.get_context_entity("active_budget")

    def get_active_category(self) -> Optional[str]:
        """Retrieve active product category."""
        return self.get_context_entity("active_category")

    def clear(self) -> None:
        """Clear all conversation history and entity state."""
        if self._session_state is not None:
            if hasattr(self._session_state, "chat_messages"):
                self._session_state.chat_messages.clear()
            if hasattr(self._session_state, "chat_context"):
                self._session_state.chat_context = {
                    "active_budget": None,
                    "active_category": None,
                    "last_products": [],
                    "active_order_id": None,
                }
        self._local_messages.clear()
        self._local_context = {
            "active_budget": None,
            "active_category": None,
            "last_products": [],
            "active_order_id": None,
        }
