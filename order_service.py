"""FastAPI Mock Order Status Service.

Provides order lookup endpoints for the e-commerce shopping assistant.
Run with: uvicorn order_service:app --reload --port 8000
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orders_data import MOCK_ORDERS, get_mock_order, list_mock_orders


class ShippingInfo(BaseModel):
    carrier: str = Field(..., description="Logistics carrier name")
    destination: str = Field(..., description="Delivery destination address or city")
    current_location: str = Field(..., description="Current checkpoint or physical location")
    last_checkpoint: str = Field(..., description="Most recently scanned logistics checkpoint")
    delivery_timestamp: Optional[str] = Field(None, description="Timestamp when delivered")


class OrderResponse(BaseModel):
    order_id: str = Field(..., description="Unique order identifier (e.g. ORD1001)")
    product_name: str = Field(..., description="Name of ordered product")
    order_status: str = Field(..., description="Status (Processing, Confirmed, Packed, Shipped, Out for delivery, Delivered)")
    order_date: str = Field(..., description="Order placement date YYYY-MM-DD")
    expected_delivery_date: str = Field(..., description="Estimated delivery date YYYY-MM-DD")
    tracking_number: str = Field(..., description="Waybill / tracking number")
    shipping_info: ShippingInfo = Field(..., description="Detailed shipping and tracking updates")


class HealthResponse(BaseModel):
    status: str
    service: str
    orders_available: int


app = FastAPI(
    title="E-Commerce Mock Order Status API",
    description="Mock order status and tracking service for the AI Shopping Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Check API health and availability."""
    return HealthResponse(
        status="healthy",
        service="order-service",
        orders_available=len(MOCK_ORDERS),
    )


@app.get("/orders", response_model=List[OrderResponse], tags=["Orders"])
def get_orders() -> List[OrderResponse]:
    """Retrieve all available mock orders for demonstration purposes."""
    return [OrderResponse(**o) for o in list_mock_orders()]


@app.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    responses={404: {"description": "Order not found"}},
    tags=["Orders"],
)
def get_order_by_id(order_id: str) -> OrderResponse:
    """Retrieve order details and tracking status by Order ID.

    Order IDs are matched case-insensitively (e.g., 'ord1002' matches 'ORD1002').
    """
    clean_id = order_id.strip()
    order = get_mock_order(clean_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order '{clean_id.upper()}' was not found in our records. Please verify the order ID (e.g., ORD1001 to ORD1007).",
        )
    return OrderResponse(**order)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("order_service:app", host="0.0.0.0", port=8000, reload=True)
