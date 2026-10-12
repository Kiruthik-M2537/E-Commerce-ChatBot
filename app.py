"""Streamlit Frontend for the AI E-Commerce Shopping Assistant (ShopMate).

Features:
- Natural language semantic product search via ChromaDB
- Direct order status tracking with FastAPI or local mock fallback
- Product comparison & conversational follow-up memory
- Polished, responsive e-commerce dashboard design
"""

import os
from typing import Any, Dict, List
import streamlit as st

from assistant import ShoppingAssistant
from memory import ConversationMemory
from order_client import default_order_client
from product_search import get_search_engine

# Configure Streamlit page
st.set_page_config(
    page_title="ShopMate — AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, elegant e-commerce design
CUSTOM_CSS = """
<style>
/* Main container padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Header Banner */
.shop-header {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
}
.shop-header h1 {
    color: white !important;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.shop-header p {
    color: #E0E7FF;
    font-size: 0.95rem;
    margin: 0;
}

/* Product Card Styling */
.product-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.product-card:hover {
    border-color: #6366F1;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.1);
    transform: translateY(-2px);
}
.product-category-badge {
    background-color: #EEF2FF;
    color: #4F46E5;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.product-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 0.35rem;
    line-height: 1.4;
}
.product-desc {
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 0.75rem;
    line-height: 1.5;
}
.product-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #F1F5F9;
    padding-top: 0.75rem;
    margin-top: 0.5rem;
}
.product-price {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
}
.product-rating {
    background-color: #FEF3C7;
    color: #92400E;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
}
.product-brand {
    font-size: 0.82rem;
    color: #64748B;
}

/* Order Status Card */
.order-card {
    background: #FFFFFF;
    border-left: 5px solid #4F46E5;
    border-top: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}
.status-badge {
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.82rem;
    display: inline-block;
}
.status-delivered { background-color: #DEF7EC; color: #03543F; }
.status-shipped { background-color: #E1EFFE; color: #1E429F; }
.status-out-for-delivery { background-color: #FEF08A; color: #854D0E; }
.status-packed { background-color: #F3E8FF; color: #6B21A8; }
.status-confirmed { background-color: #DBEAFE; color: #1E40AF; }
.status-processing { background-color: #F1F5F9; color: #334155; }

/* Status pill indicator in sidebar */
.api-badge-live {
    background-color: #DEF7EC;
    color: #03543F;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
.api-badge-fallback {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_context" not in st.session_state:
    st.session_state.chat_context = {
        "active_budget": None,
        "active_category": None,
        "last_products": [],
        "active_order_id": None,
    }

# Initialize Memory and Assistant
memory = ConversationMemory(session_state=st.session_state)
search_engine = get_search_engine()
assistant = ShoppingAssistant(memory=memory, search_engine=search_engine, order_client=default_order_client)


def render_product_card(product: Dict[str, Any]) -> None:
    """Render a structured e-commerce product recommendation card."""
    price = float(product.get("price", 0))
    rating = product.get("rating", "4.0")
    category = product.get("category", "General").title()
    brand = product.get("brand", "Generic")
    name = product.get("name", "Product")
    desc = product.get("description", "")
    url = product.get("url", "#")

    st.markdown(
        f"""
        <div class="product-card">
            <div>
                <span class="product-category-badge">{category}</span>
                <span class="product-brand">• {brand}</span>
            </div>
            <div class="product-title">{name}</div>
            <div class="product-desc">{desc}</div>
            <div class="product-footer">
                <div>
                    <span class="product-price">₹{price:,.2f}</span>
                </div>
                <div>
                    <span class="product-rating">★ {rating}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Action button for catalog view (avoiding misleading real purchase links)
    if url and url != "#":
        st.markdown(
            f"<a href='{url}' target='_blank' style='font-size: 0.8rem; color: #4F46E5; text-decoration: none; font-weight: 500;'>🔗 View Catalog Item ({product.get('id', '')}) ↗</a>",
            unsafe_allow_html=True,
        )


def render_order_card(order: Dict[str, Any]) -> None:
    """Render a structured order status tracking card."""
    status = order.get("order_status", "Processing")
    order_id = order.get("order_id", "N/A")
    product_name = order.get("product_name", "N/A")
    order_date = order.get("order_date", "N/A")
    delivery_date = order.get("expected_delivery_date", "N/A")
    tracking_no = order.get("tracking_number", "N/A")
    shipping = order.get("shipping_info", {})
    carrier = shipping.get("carrier", "Standard Carrier")
    location = shipping.get("current_location", "In Transit")
    destination = shipping.get("destination", "Customer Address")

    status_class_map = {
        "Delivered": "status-delivered",
        "Shipped": "status-shipped",
        "Out for delivery": "status-out-for-delivery",
        "Packed": "status-packed",
        "Confirmed": "status-confirmed",
        "Processing": "status-processing",
    }
    status_class = status_class_map.get(status, "status-processing")

    st.markdown(
        f"""
        <div class="order-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">Order #{order_id}</span>
                <span class="status-badge {status_class}">● {status}</span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #1E293B; margin-bottom: 0.5rem;">
                📦 {product_name}
            </div>
            <div style="font-size: 0.85rem; color: #475569; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.75rem; border-top: 1px solid #F1F5F9; padding-top: 0.75rem;">
                <div><strong>Order Date:</strong> {order_date}</div>
                <div><strong>Expected Delivery:</strong> {delivery_date}</div>
                <div><strong>Carrier:</strong> {carrier}</div>
                <div><strong>Tracking #:</strong> <code>{tracking_no}</code></div>
                <div><strong>Destination:</strong> {destination}</div>
                <div><strong>Current Location:</strong> {location}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛍️ ShopMate Assistant")
    st.markdown("Your AI assistant for product discovery, comparisons, and order tracking.")
    st.markdown("---")

    # API Health Check
    api_online = default_order_client.is_api_reachable()
    if api_online:
        st.markdown(
            "**Order API Status:** <span class='api-badge-live'>🟢 Live FastAPI (Port 8000)</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "**Order API Status:** <span class='api-badge-fallback'>🟡 Offline Fallback Active</span>",
            unsafe_allow_html=True,
        )
        st.caption("FastAPI is not detected on port 8000. Fallback data is used automatically for orders.")

    # LLM Mode Status
    if assistant.is_llm_enabled:
        st.markdown(f"**AI Engine:** `OpenAI ({assistant.model_name})`")
    else:
        st.markdown("**AI Engine:** `Deterministic Engine (Offline Ready)`")
        st.caption("Running in zero-cost offline mode without API keys required.")

    st.markdown("---")

    # Suggested Prompts
    st.markdown("#### 💡 Try Asking:")
    example_prompts = [
        "Find wireless headphones under ₹3000",
        "Which one has the highest rating?",
        "Show me cheaper alternatives",
        "Where is my order ORD1002?",
        "Recommend a gift for a student",
        "Show mechanical keyboards",
        "Track order ORD1003",
    ]

    for p in example_prompts:
        if st.button(p, key=f"btn_{p}", use_container_width=True):
            st.session_state["pending_input"] = p

    st.markdown("---")

    # Sample Order IDs Cheat Sheet
    with st.expander("📦 Sample Orders Reference"):
        st.markdown("""
        - **ORD1001**: boAt Headphones (*Delivered*)
        - **ORD1002**: Sony WH-1000XM4 (*Shipped*)
        - **ORD1003**: Keychron Keyboard (*Out for delivery*)
        - **ORD1004**: Apple iPad (*Packed*)
        - **ORD1005**: Portronics Mouse (*Confirmed*)
        - **ORD1006**: JBL Speaker (*Processing*)
        - **ORD1007**: Smartwatch (*Delivered*)
        """)

    # Clear Chat Button
    if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
        memory.clear()
        st.rerun()


# --- MAIN HEADER ---
st.markdown(
    """
    <div class="shop-header">
        <h1>🛍️ ShopMate AI Shopping Assistant</h1>
        <p>Explore products with natural language, enforce budget limits, compare ratings, and track live shipments.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Chat History
messages = memory.get_messages()

if not messages:
    # Empty State Welcome
    st.info(
        "👋 **Welcome! How can I help you shop today?**\n\n"
        "You can search for products (e.g. *'Show me speakers under ₹2500'*), "
        "ask follow-up questions to compare options, or track any package using your Order ID (e.g. *'Where is my order ORD1002?'*)."
    )
else:
    for msg in messages:
        if msg.role == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg.content)
        elif msg.role == "assistant":
            with st.chat_message("assistant", avatar="🛍️"):
                st.markdown(msg.content)

                # Render structured product cards if attached to this turn
                if msg.products:
                    st.markdown("##### 🛒 Recommended Products:")
                    cols = st.columns(min(len(msg.products), 2))
                    for idx, prod in enumerate(msg.products):
                        with cols[idx % 2]:
                            render_product_card(prod)

                # Render structured order card if attached to this turn
                if msg.order:
                    st.markdown("##### 📦 Order Tracking Details:")
                    render_order_card(msg.order)


# Handle Input (from chat_input or sidebar prompt button)
user_input = st.chat_input("Ask for product recommendations, compare items, or enter an Order ID...")

if "pending_input" in st.session_state and st.session_state["pending_input"]:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # Process message through the coordinator
    result = assistant.process_message(user_input)
    st.rerun()
