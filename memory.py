"""Conversational Memory System for AI Shopping Assistant.

Manages message history, turn windowing, active entities (budget, category, products, orders),
and integrates with Streamlit session_state while remaining fully usable in headless unit tests.
Provides complete conversation history tracking, session export (JSON & Markdown), and analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    products: Optional[List[Dict[str, Any]]] = None
    order: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_time(self) -> str:
        """Return clean human-readable timestamp (e.g. '10:45 AM' or 'Oct 12, 10:45 AM')."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%I:%M %p")
        except Exception:
            return self.timestamp[:16] if self.timestamp else ""

    @property
    def formatted_date_time(self) -> str:
        """Return full formatted date and time (e.g. 'Oct 12, 2026 • 10:45 AM')."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%b %d, %Y • %I:%M %p")
        except Exception:
            return self.timestamp or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time,
            "products": self.products,
            "order": self.order,
            "metadata": self.metadata,
        }


class ConversationMemory:
    """Session-level conversation memory with entity tracking, turn windowing, and history export."""

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

    def get_history_turns(self) -> List[Dict[str, Any]]:
        """Group chronological messages into paired turns for history exploration."""
        turns: List[Dict[str, Any]] = []
        msgs = self.messages
        current_turn: Optional[Dict[str, Any]] = None

        for msg in msgs:
            if msg.role == "user":
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {
                    "turn_index": len(turns) + 1,
                    "user_query": msg.content,
                    "user_time": msg.formatted_time,
                    "user_full_time": msg.formatted_date_time,
                    "assistant_response": "",
                    "assistant_time": "",
                    "intent": "general",
                    "products": [],
                    "order": None,
                }
            elif msg.role == "assistant":
                if current_turn is None:
                    current_turn = {
                        "turn_index": len(turns) + 1,
                        "user_query": "Initial Conversation",
                        "user_time": msg.formatted_time,
                        "user_full_time": msg.formatted_date_time,
                        "assistant_response": msg.content,
                        "assistant_time": msg.formatted_time,
                        "intent": msg.metadata.get("intent", "general"),
                        "products": msg.products or [],
                        "order": msg.order,
                    }
                else:
                    current_turn["assistant_response"] = msg.content
                    current_turn["assistant_time"] = msg.formatted_time
                    current_turn["intent"] = msg.metadata.get("intent", "general")
                    current_turn["products"] = msg.products or []
                    current_turn["order"] = msg.order
                turns.append(current_turn)
                current_turn = None

        if current_turn is not None:
            turns.append(current_turn)

        return turns

    def get_session_stats(self) -> Dict[str, Any]:
        """Calculate shopping session analytics."""
        msgs = self.messages
        user_msgs = [m for m in msgs if m.role == "user"]
        assistant_msgs = [m for m in msgs if m.role == "assistant"]

        all_products: List[Dict[str, Any]] = []
        orders_tracked: List[str] = []
        categories_seen = set()

        for m in assistant_msgs:
            if m.products:
                for p in m.products:
                    all_products.append(p)
                    if "category" in p:
                        categories_seen.add(p["category"])
            if m.order and "order_id" in m.order:
                orders_tracked.append(m.order["order_id"])

        return {
            "total_messages": len(msgs),
            "total_turns": len(user_msgs),
            "products_recommended_count": len(all_products),
            "unique_categories": sorted(list(categories_seen)),
            "orders_tracked": sorted(list(set(orders_tracked))),
            "active_budget": self.get_active_budget(),
            "active_category": self.get_active_category(),
        }

    def export_as_json(self) -> str:
        """Export complete conversation history as formatted JSON."""
        data = {
            "session_exported_at": datetime.now().isoformat(),
            "stats": self.get_session_stats(),
            "context": self.context,
            "turns": self.get_history_turns(),
            "raw_messages": [m.to_dict() for m in self.messages],
        }
        return json.dumps(data, indent=2)

    def export_as_markdown(self) -> str:
        """Export conversation history as an elegant markdown document."""
        stats = self.get_session_stats()
        lines = [
            "# 🛍️ ShopMate Shopping Assistant — Conversation Transcript",
            f"**Exported:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            "",
            "## 📊 Session Summary",
            f"- **Total Interactions:** {stats['total_turns']} turns ({stats['total_messages']} messages)",
            f"- **Products Recommended:** {stats['products_recommended_count']}",
            f"- **Categories Explored:** {', '.join(stats['unique_categories']) if stats['unique_categories'] else 'None'}",
            f"- **Orders Tracked:** {', '.join(stats['orders_tracked']) if stats['orders_tracked'] else 'None'}",
            f"- **Active Budget:** ₹{stats['active_budget']:,.2f}" if stats['active_budget'] else "- **Active Budget:** None",
            "",
            "---",
            "## 💬 Conversation History",
            "",
        ]

        turns = self.get_history_turns()
        if not turns:
            lines.append("*No messages recorded in this session.*")
        else:
            for turn in turns:
                lines.append(f"### Turn #{turn['turn_index']} • {turn['user_time']}")
                lines.append(f"**👤 You:** {turn['user_query']}")
                lines.append("")
                lines.append(f"**🛍️ ShopMate ({turn['intent'].replace('_', ' ').title()}):**")
                lines.append(turn['assistant_response'])
                lines.append("")

                if turn.get("products"):
                    lines.append("**Recommended Items:**")
                    for p in turn["products"]:
                        price = f"₹{float(p.get('price', 0)):,.2f}"
                        lines.append(f"- **{p.get('name')}** — {price} (★ {p.get('rating', '4.0')}) | *{p.get('brand')}*")
                    lines.append("")

                if turn.get("order"):
                    ord_data = turn["order"]
                    lines.append(f"**Tracked Order #{ord_data.get('order_id')}:** Status: `{ord_data.get('order_status')}` | Expected: {ord_data.get('expected_delivery_date')}")
                    lines.append("")

                lines.append("---")

        return "\n".join(lines)

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
