"""Tests for Intent Routing and Entity Extraction."""

import pytest
from intent_router import IntentRouter


@pytest.fixture
def router():
    return IntentRouter()


def test_order_tracking_intent_detection(router):
    """Verify order tracking intent and ID extraction."""
    res1 = router.route("Where is my order ORD1002?")
    assert res1.intent == "order_tracking"
    assert res1.order_id == "ORD1002"

    res2 = router.route("Please track order ord-1003")
    assert res2.intent == "order_tracking"
    assert res2.order_id == "ORD1003"

    res3 = router.route("Check delivery status for ORD1007")
    assert res3.intent == "order_tracking"
    assert res3.order_id == "ORD1007"


def test_product_search_intent_and_budget_extraction(router):
    """Verify natural language search intent, budget parsing, and category extraction."""
    res1 = router.route("Find wireless headphones under ₹3000")
    assert res1.intent == "product_search"
    assert res1.max_price == 3000.0
    assert res1.category == "audio"

    res2 = router.route("Show me a speaker below 2500")
    assert res2.intent == "product_search"
    assert res2.max_price == 2500.0
    assert res2.category == "audio"

    res3 = router.route("I need a laptop accessory for studying")
    assert res3.intent == "product_search"
    assert res3.category == "accessories"


def test_product_comparison_intent(router):
    """Verify comparison intent when previous products exist in context."""
    context_with_products = {
        "last_products": [
            {"name": "boAt Rockerz 450", "price": 1499, "rating": 4.2},
            {"name": "OnePlus Bullets Z2", "price": 1799, "rating": 4.3},
        ]
    }

    res1 = router.route("Which one has the highest rating?", conversation_context=context_with_products)
    assert res1.intent == "product_comparison"
    assert res1.comparison_metric == "rating"

    res2 = router.route("Which one is cheaper?", conversation_context=context_with_products)
    assert res2.intent == "product_comparison"
    assert res2.comparison_metric == "price_low"


def test_follow_up_intent(router):
    """Verify follow-up intent for cheaper alternatives."""
    context_with_products = {
        "last_products": [
            {"name": "Sony WH-1000XM4", "price": 19990, "category": "audio"},
            {"name": "Apple AirPods Pro", "price": 24900, "category": "audio"},
        ],
        "active_category": "audio",
    }

    res = router.route("Show me cheaper alternatives", conversation_context=context_with_products)
    assert res.intent == "follow_up"
    assert res.category == "audio"
    assert res.max_price is not None
    assert res.max_price < 19990.0


def test_greeting_intent(router):
    """Verify basic greetings are classified as greeting_help."""
    res1 = router.route("Hello!")
    assert res1.intent == "greeting_help"

    res2 = router.route("What can you do?")
    assert res2.intent == "greeting_help"
