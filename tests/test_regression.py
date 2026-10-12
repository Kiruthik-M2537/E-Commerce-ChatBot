"""Regression tests covering all 12 audited requirements.

1. Greeting does not call ChromaDB.
2. Greeting does not render/attach product cards.
3. Product-search intent retrieves catalog results.
4. Budget constraints are respected.
5. Order tracking calls the order API.
6. Invalid order IDs produce appropriate errors.
7. Follow-up comparison uses the previous search results.
8. Unrelated messages do not reuse stale product results.
9. Clearing conversation resets memory and product context.
10. Gemini failures trigger the correct intent-specific fallback.
11. Streamlit session memory updates do not duplicate assistant responses.
12. Missing API credentials do not crash the application.
"""

from unittest.mock import MagicMock, patch
import pytest

from assistant import ShoppingAssistant
from intent_router import (
    INTENT_GREETING,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_COMPARISON,
    INTENT_ORDER_TRACKING,
    INTENT_FOLLOW_UP,
    INTENT_GENERAL_QUESTION,
    IntentRouter,
)
from memory import ConversationMemory
from order_client import OrderClient
from product_search import ProductSearchEngine, get_search_engine


@pytest.fixture
def memory():
    return ConversationMemory()


@pytest.fixture
def search_engine():
    return get_search_engine()


@pytest.fixture
def assistant(memory, search_engine):
    # Standalone assistant without external API dependencies for deterministic regression testing
    with patch.dict("os.environ", {"GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}):
        return ShoppingAssistant(memory=memory, search_engine=search_engine)


def test_1_and_2_greeting_does_not_call_chromadb_or_attach_products(assistant, search_engine):
    """1 & 2: Greeting must not invoke ChromaDB search and must not attach product cards."""
    with patch.object(search_engine, "search_products", wraps=search_engine.search_products) as mock_search:
        res = assistant.process_message("how are you")
        assert res["intent"] == INTENT_GREETING
        assert len(res["products"]) == 0
        assert res["order"] is None
        assert "found" not in res["response_text"].lower()
        # Ensure ChromaDB search was never called
        mock_search.assert_not_called()

        res2 = assistant.process_message("Hi, how are you?")
        assert res2["intent"] == INTENT_GREETING
        assert len(res2["products"]) == 0
        mock_search.assert_not_called()


def test_3_product_search_retrieves_catalog_results(assistant):
    """3: Product search intent retrieves actual matching catalog products."""
    res = assistant.process_message("Find wireless headphones under ₹3000")
    assert res["intent"] == INTENT_PRODUCT_SEARCH
    assert len(res["products"]) > 0
    assert any("boat" in p["name"].lower() or "oneplus" in p["name"].lower() for p in res["products"])


def test_4_budget_constraints_are_strictly_respected(assistant):
    """4: Budget constraints are strictly enforced with zero products over max price."""
    max_budget = 3000.0
    res = assistant.process_message("Find wireless headphones under ₹3000")
    for prod in res["products"]:
        assert float(prod["price"]) <= max_budget


def test_5_order_tracking_calls_order_api(assistant):
    """5: Order tracking calls the order client and returns verified details."""
    with patch.object(assistant.order_client, "get_order", wraps=assistant.order_client.get_order) as mock_order:
        res = assistant.process_message("Where is my order ORD1002?")
        assert res["intent"] == INTENT_ORDER_TRACKING
        mock_order.assert_called_once_with("ORD1002")
        assert res["order"] is not None
        assert res["order"]["order_id"] == "ORD1002"
        assert res["order"]["order_status"] == "Shipped"


def test_6_invalid_order_id_produces_appropriate_error(assistant):
    """6: Unknown order ID returns a helpful error instead of fabricating details."""
    res = assistant.process_message("Where is my order ORD99999?")
    assert res["intent"] == INTENT_ORDER_TRACKING
    assert res["order"] is None
    assert "not found" in res["response_text"].lower()
    assert "ORD1001" in res["response_text"]


def test_7_follow_up_comparison_uses_previous_results(assistant):
    """7: Follow-up comparison queries compare the previously discussed items."""
    # First search for headphones
    assistant.process_message("Find wireless headphones under ₹3000")

    # Follow-up: ask which has the highest rating
    res = assistant.process_message("Which one has the highest rating?")
    assert res["intent"] == INTENT_PRODUCT_COMPARISON
    assert len(res["products"]) > 0
    # Should identify the highest rated product among them
    assert "highest" in res["response_text"].lower() or "rating" in res["response_text"].lower()


def test_8_unrelated_messages_do_not_reuse_stale_product_results(assistant):
    """8: An unrelated greeting or general question does not attach previous product results."""
    # Step 1: User searches products
    assistant.process_message("Find wireless headphones under ₹3000")

    # Step 2: User says thanks
    res = assistant.process_message("Thanks, that's helpful.")
    assert res["intent"] == INTENT_GREETING
    # Crucial: This turn's product list must be empty
    assert len(res["products"]) == 0
    assert res["order"] is None


def test_9_clearing_conversation_resets_memory_and_product_context(assistant):
    """9: Clearing conversation resets all memory and context entities."""
    assistant.process_message("Find speakers under ₹2500")
    assert len(assistant.memory.get_last_recommended_products()) > 0

    assistant.memory.clear()
    assert len(assistant.memory.get_messages()) == 0
    assert len(assistant.memory.get_last_recommended_products()) == 0
    assert assistant.memory.get_active_budget() is None
    assert assistant.memory.get_active_category() is None

    # After clearing, comparison query knows nothing has been discussed
    res = assistant.process_message("Which one has the highest rating?")
    assert "don't have any previously recommended products" in res["response_text"].lower()


def test_10_gemini_failure_triggers_correct_intent_specific_fallback(assistant):
    """10: API errors (e.g. 503/timeout) trigger intent-specific fallbacks without crashing."""
    # Mock LLM failure
    with patch.object(assistant, "_call_llm", return_value=None):
        # Greeting fallback
        res_greet = assistant.process_message("hello")
        assert res_greet["intent"] == INTENT_GREETING
        assert "ShopMate" in res_greet["response_text"]

        # Search fallback
        res_search = assistant.process_message("find mechanical keyboards")
        assert res_search["intent"] == INTENT_PRODUCT_SEARCH
        assert len(res_search["products"]) > 0

        # Order fallback
        res_order = assistant.process_message("track ORD1001")
        assert res_order["intent"] == INTENT_ORDER_TRACKING
        assert res_order["order"]["order_status"] == "Delivered"


def test_11_session_state_updates_do_not_duplicate_responses(memory):
    """11: Message additions produce distinct historical entries without duplication."""
    memory.add_user_message("Query 1")
    memory.add_assistant_message("Answer 1")
    memory.add_user_message("Query 2")
    memory.add_assistant_message("Answer 2")

    messages = memory.get_messages()
    assert len(messages) == 4
    contents = [m.content for m in messages]
    assert contents == ["Query 1", "Answer 1", "Query 2", "Answer 2"]


def test_12_missing_api_credentials_do_not_crash_application(search_engine):
    """12: Application operates seamlessly when zero API credentials are provided."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
        offline_assistant = ShoppingAssistant(search_engine=search_engine)
        assert offline_assistant.is_llm_enabled is False
        assert "Deterministic" in offline_assistant.engine_name

        res = offline_assistant.process_message("how are you")
        assert res["intent"] == INTENT_GREETING
        assert len(res["products"]) == 0
