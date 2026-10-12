"""Client for interacting with the FastAPI Order Status Service.

Includes health checks, HTTP requests, and a robust fallback mechanism
for when the FastAPI backend is running offline or not reachable.
"""

import os
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

from orders_data import get_mock_order, list_mock_orders

load_dotenv()

DEFAULT_ORDER_API_URL = "http://127.0.0.1:8000"


class OrderClient:
    """Client for retrieving order tracking information with live API and local fallback."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 3.0):
        self.base_url = (base_url or os.getenv("ORDER_API_URL", DEFAULT_ORDER_API_URL)).rstrip("/")
        self.timeout = timeout

    def is_api_reachable(self) -> bool:
        """Check if the live FastAPI order service is reachable via HTTP."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Fetch order details by ID from live FastAPI service or fallback to local mock data.

        Returns a dictionary with:
        - success: bool
        - order: dict or None
        - error: str or None
        - source: 'live_api' | 'local_fallback' | 'not_found'
        - is_fallback: bool
        """
        clean_id = (order_id or "").strip().upper()
        if not clean_id:
            return {
                "success": False,
                "order": None,
                "error": "No order ID was provided.",
                "source": "invalid_input",
                "is_fallback": False,
            }

        # Attempt Live HTTP API call first
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/orders/{clean_id}")
                if response.status_code == 200:
                    order_data = response.json()
                    return {
                        "success": True,
                        "order": order_data,
                        "error": None,
                        "source": "live_api",
                        "is_fallback": False,
                    }
                elif response.status_code == 404:
                    detail = response.json().get("detail", f"Order {clean_id} not found.")
                    return {
                        "success": False,
                        "order": None,
                        "error": detail,
                        "source": "live_api",
                        "is_fallback": False,
                    }
                else:
                    # Unexpected status code, drop down to fallback
                    pass
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            # Live API unavailable; use local mock fallback
            pass

        # Local Fallback Mode
        local_order = get_mock_order(clean_id)
        if local_order:
            return {
                "success": True,
                "order": local_order,
                "error": None,
                "source": "local_fallback",
                "is_fallback": True,
            }
        else:
            return {
                "success": False,
                "order": None,
                "error": f"Order '{clean_id}' was not found in our records (tested against fallback catalog).",
                "source": "local_fallback",
                "is_fallback": True,
            }

    def list_all_orders(self) -> List[Dict[str, Any]]:
        """List all mock orders, querying live API if possible, else falling back."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/orders")
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return list_mock_orders()


# Default singleton instance for easy import
default_order_client = OrderClient()
