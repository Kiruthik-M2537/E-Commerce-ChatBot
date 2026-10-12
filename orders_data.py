"""Centralized mock order database for the e-commerce shopping assistant."""

from typing import Dict, Any, List

MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD1001": {
        "order_id": "ORD1001",
        "product_name": "boAt Rockerz 450 Bluetooth On-Ear Headphones",
        "order_status": "Delivered",
        "order_date": "2026-09-28",
        "expected_delivery_date": "2026-10-02",
        "tracking_number": "TRK-IN-9823101",
        "shipping_info": {
            "carrier": "BlueDart Express",
            "destination": "Bengaluru, Karnataka",
            "current_location": "Delivered to Customer (Front Door)",
            "last_checkpoint": "Bengaluru Delivery Hub",
            "delivery_timestamp": "2026-10-02 14:35 IST",
        },
    },
    "ORD1002": {
        "order_id": "ORD1002",
        "product_name": "Sony WH-1000XM4 Wireless Noise-Cancelling Headphones",
        "order_status": "Shipped",
        "order_date": "2026-10-08",
        "expected_delivery_date": "2026-10-14",
        "tracking_number": "TRK-IN-4412002",
        "shipping_info": {
            "carrier": "Delhivery Priority",
            "destination": "Mumbai, Maharashtra",
            "current_location": "In Transit - Pune Sorting Facility",
            "last_checkpoint": "Departed Regional Sorting Center Pune",
            "delivery_timestamp": None,
        },
    },
    "ORD1003": {
        "order_id": "ORD1003",
        "product_name": "Keychron K2 V2 Wireless Mechanical Keyboard",
        "order_status": "Out for delivery",
        "order_date": "2026-10-09",
        "expected_delivery_date": "2026-10-12",
        "tracking_number": "TRK-IN-7789003",
        "shipping_info": {
            "carrier": "Ekart Logistics",
            "destination": "Chennai, Tamil Nadu",
            "current_location": "Out with Delivery Agent (Arun K.)",
            "last_checkpoint": "Dispatched from T. Nagar Hub",
            "delivery_timestamp": None,
        },
    },
    "ORD1004": {
        "order_id": "ORD1004",
        "product_name": "Apple iPad 10th Gen 64GB Wi-Fi",
        "order_status": "Packed",
        "order_date": "2026-10-10",
        "expected_delivery_date": "2026-10-16",
        "tracking_number": "TRK-IN-3321004",
        "shipping_info": {
            "carrier": "BlueDart Express",
            "destination": "Hyderabad, Telangana",
            "current_location": "Fulfillment Center Warehouse - Ready for Pickup",
            "last_checkpoint": "Package Sealed and Invoiced",
            "delivery_timestamp": None,
        },
    },
    "ORD1005": {
        "order_id": "ORD1005",
        "product_name": "Portronics Toad 23 Wireless Optical Mouse",
        "order_status": "Confirmed",
        "order_date": "2026-10-11",
        "expected_delivery_date": "2026-10-17",
        "tracking_number": "TRK-IN-1122005",
        "shipping_info": {
            "carrier": "Shadowfax Logistics",
            "destination": "Delhi NCR",
            "current_location": "Order Processing Center",
            "last_checkpoint": "Payment Verified & Inventory Allocated",
            "delivery_timestamp": None,
        },
    },
    "ORD1006": {
        "order_id": "ORD1006",
        "product_name": "JBL Flip 6 Portable Bluetooth Speaker",
        "order_status": "Processing",
        "order_date": "2026-10-12",
        "expected_delivery_date": "2026-10-18",
        "tracking_number": "TRK-IN-5566006",
        "shipping_info": {
            "carrier": "FedEx India",
            "destination": "Kolkata, West Bengal",
            "current_location": "Seller Warehouse",
            "last_checkpoint": "Order received by seller",
            "delivery_timestamp": None,
        },
    },
    "ORD1007": {
        "order_id": "ORD1007",
        "product_name": "Noise ColorFit Pro 4 Alpha Smartwatch",
        "order_status": "Delivered",
        "order_date": "2026-09-25",
        "expected_delivery_date": "2026-09-29",
        "tracking_number": "TRK-IN-6677007",
        "shipping_info": {
            "carrier": "Delhivery Priority",
            "destination": "Pune, Maharashtra",
            "current_location": "Delivered to Reception / Security",
            "last_checkpoint": "Pune East Hub",
            "delivery_timestamp": "2026-09-29 11:20 IST",
        },
    },
}


def get_mock_order(order_id: str) -> Dict[str, Any] | None:
    """Retrieve a mock order by ID (case-insensitive)."""
    if not order_id:
        return None
    normalized_id = order_id.strip().upper()
    return MOCK_ORDERS.get(normalized_id)


def list_mock_orders() -> List[Dict[str, Any]]:
    """List all available mock orders."""
    return list(MOCK_ORDERS.values())
