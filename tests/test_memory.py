"""Tests for ConversationMemory and entity tracking."""

import pytest
from memory import ConversationMemory


def test_add_and_retrieve_messages():
    """Verify adding user and assistant messages and retrieving history."""
    mem = ConversationMemory(max_turns=5)
    mem.add_user_message("I need wireless headphones under 3000")
    mem.add_assistant_message(
        "Here are matching headphones:",
        products=[{"name": "boAt Rockerz 450", "price": 1499, "category": "audio"}],
    )

    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert len(msgs[1].products) == 1
    assert mem.get_active_category() == "audio"


def test_turn_windowing_limits_unbounded_growth():
    """Verify memory restricts historical messages to 2 * max_turns."""
    mem = ConversationMemory(max_turns=2)  # Max 4 total messages

    for i in range(5):
        mem.add_user_message(f"User message {i}")
        mem.add_assistant_message(f"Assistant response {i}")

    msgs = mem.get_messages()
    assert len(msgs) == 4  # 2 turns * 2 = 4 messages
    assert msgs[-1].content == "Assistant response 4"
    assert msgs[0].content == "User message 3"


def test_context_entity_preservation():
    """Verify preserving budget, category, and previous products."""
    mem = ConversationMemory()
    mem.set_context_entity("active_budget", 2500.0)
    mem.set_context_entity("active_category", "accessories")

    dummy_products = [{"id": "PROD107", "name": "Logitech MX Master", "price": 8495.0}]
    mem.add_assistant_message("Check out this mouse", products=dummy_products)

    assert mem.get_active_budget() == 2500.0
    assert mem.get_active_category() == "accessories"
    assert len(mem.get_last_recommended_products()) == 1
    assert mem.get_last_recommended_products()[0]["name"] == "Logitech MX Master"


def test_memory_clear():
    """Verify clearing conversation resets both messages and context entities."""
    mem = ConversationMemory()
    mem.add_user_message("Hello")
    mem.set_context_entity("active_budget", 5000.0)
    mem.set_context_entity("active_order_id", "ORD1002")

    mem.clear()
    assert len(mem.get_messages()) == 0
    assert mem.get_active_budget() is None
    assert mem.get_context_entity("active_order_id") is None
    assert len(mem.get_last_recommended_products()) == 0
