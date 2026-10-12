"""Tests for FastAPI Order Service and OrderClient."""

import pytest
from starlette.testclient import TestClient

from order_service import app
from order_client import OrderClient


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_health_check_endpoint(api_client):
    """Verify GET /health returns 200 with service info."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "order-service"
    assert data["orders_available"] >= 6


def test_list_orders_endpoint(api_client):
    """Verify GET /orders returns the mock orders list."""
    response = api_client.get("/orders")
    assert response.status_code == 200
    orders = response.json()
    assert isinstance(orders, list)
    assert len(orders) >= 6

    # Verify all 6 required statuses exist across the dataset
    statuses = {o["order_status"] for o in orders}
    expected_statuses = {"Processing", "Confirmed", "Packed", "Shipped", "Out for delivery", "Delivered"}
    assert expected_statuses.issubset(statuses)


def test_get_valid_order_exact_case(api_client):
    """Verify fetching valid order with exact uppercase ID."""
    response = api_client.get("/orders/ORD1001")
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD1001"
    assert data["order_status"] == "Delivered"
    assert "boAt Rockerz 450" in data["product_name"]
    assert data["shipping_info"]["carrier"] == "BlueDart Express"


def test_get_valid_order_case_insensitive(api_client):
    """Verify order ID lookup is case-insensitive (e.g. ord1002)."""
    response = api_client.get("/orders/ord1002")
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD1002"
    assert data["order_status"] == "Shipped"


def test_get_invalid_order_returns_404(api_client):
    """Verify unknown order returns HTTP 404 with helpful error."""
    response = api_client.get("/orders/ORD99999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "ORD99999" in data["detail"]


def test_order_client_local_fallback_when_api_unreachable():
    """Verify OrderClient transparently falls back to local data when API is down."""
    # Point to non-existent port to simulate unreachable server
    offline_client = OrderClient(base_url="http://127.0.0.1:59999", timeout=0.5)

    assert offline_client.is_api_reachable() is False

    # Lookup valid order through fallback
    result = offline_client.get_order("ORD1003")
    assert result["success"] is True
    assert result["is_fallback"] is True
    assert result["source"] == "local_fallback"
    assert result["order"]["order_id"] == "ORD1003"
    assert result["order"]["order_status"] == "Out for delivery"


def test_order_client_unknown_order_fallback():
    """Verify OrderClient handles unknown orders gracefully in fallback mode."""
    offline_client = OrderClient(base_url="http://127.0.0.1:59999", timeout=0.5)
    result = offline_client.get_order("NON_EXISTENT_999")
    assert result["success"] is False
    assert result["order"] is None
    assert result["is_fallback"] is True
    assert "not found" in result["error"].lower()
