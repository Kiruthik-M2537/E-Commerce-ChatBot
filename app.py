"""Streamlit Frontend for the AI E-Commerce Shopping Assistant (ShopMate).

Delivers a high-conversion, professional e-commerce experience inspired by Amazon and Flipkart:
- Modern header with brand identity, live status, location, and quick category pills
- Rich product cards with Assured/Prime badges, strikethrough MRP, ratings, and Add to Cart
- Amazon/Flipkart-style 5-step visual order tracking stepper
- Comprehensive conversation history drawer, turn-by-turn audit, and JSON/Markdown export
- Interactive shopping cart simulation and full product catalog explorer
"""

import importlib
import os
from typing import Any, Dict, List, Optional
import streamlit as st

import memory as memory_module
try:
    importlib.reload(memory_module)
except Exception:
    pass

from assistant import ShoppingAssistant
from memory import ConversationMemory
from order_client import default_order_client
from product_search import get_search_engine

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopMate — AI Shopping Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Amazon & Flipkart Inspired Custom CSS
# -----------------------------------------------------------------------------
AMAZON_FLIPKART_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

/* Typography & Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}

/* Main Container */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1280px;
}

/* Amazon / Flipkart Top Brand Bar */
.ecommerce-topbar {
    background: linear-gradient(90deg, #131921 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 12px;
    padding: 1.1rem 1.6rem;
    color: #FFFFFF;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.25);
    border-bottom: 3px solid #FF9900;
}
.brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.brand-tag {
    background: linear-gradient(135deg, #FF9900 0%, #F59E0B 100%);
    color: #111827;
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    letter-spacing: 0.04em;
    margin-left: 0.4rem;
}
.assured-pill {
    background: #0284C7;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}
.topbar-subtext {
    color: #94A3B8;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

/* Category Pills Bar */
.category-pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.5rem 0;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid #E2E8F0;
}

/* Product Card (Amazon / Flipkart Style) */
.amz-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}
.amz-card:hover {
    border-color: #2874F0;
    box-shadow: 0 8px 24px rgba(40, 116, 240, 0.12);
    transform: translateY(-2px);
}
.card-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.badge-category {
    background: #F1F5F9;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
}
.badge-deal {
    background: #CC0C39;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.25rem 0.55rem;
    border-radius: 4px;
    letter-spacing: 0.02em;
}
.badge-assured {
    background: #2874F0;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
}
.product-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.35;
    margin-bottom: 0.4rem;
}
.product-brand {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.product-desc {
    font-size: 0.86rem;
    color: #475569;
    line-height: 1.45;
    margin-bottom: 0.85rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.rating-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.rating-pill {
    background: #388E3C;
    color: #FFFFFF;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
}
.reviews-count {
    color: #007185;
    font-size: 0.82rem;
    font-weight: 500;
}
.bought-badge {
    color: #565959;
    font-size: 0.78rem;
    background: #F8FAFC;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}
.pricing-block {
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid #F1F5F9;
}
.current-price {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    display: inline-block;
    margin-right: 0.5rem;
}
.mrp-price {
    font-size: 0.88rem;
    color: #565959;
    text-decoration: line-through;
    margin-right: 0.5rem;
}
.discount-tag {
    color: #CC0C39;
    font-weight: 700;
    font-size: 0.9rem;
}
.delivery-promise {
    font-size: 0.82rem;
    color: #0F172A;
    margin-bottom: 0.35rem;
}
.delivery-promise strong {
    color: #007600;
}
.stock-tag {
    font-size: 0.78rem;
    font-weight: 600;
    color: #007600;
    margin-bottom: 0.75rem;
}

/* Amazon / Flipkart Order Tracking Stepper */
.amz-order-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}
.order-header-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.9rem;
    margin-bottom: 1.2rem;
}
.order-id-label {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #0F172A;
}
.stepper-container {
    margin: 1.4rem 0;
    position: relative;
}
.stepper-progress-track {
    height: 6px;
    background: #E2E8F0;
    border-radius: 9999px;
    position: absolute;
    top: 14px;
    left: 4%;
    right: 4%;
    z-index: 1;
}
.stepper-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #2874F0 0%, #10B981 100%);
    border-radius: 9999px;
    transition: width 0.4s ease;
}
.stepper-nodes {
    display: flex;
    justify-content: space-between;
    position: relative;
    z-index: 2;
}
.stepper-node {
    text-align: center;
    width: 20%;
}
.node-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #FFFFFF;
    border: 3px solid #CBD5E1;
    color: #64748B;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 0 auto 0.4rem auto;
    transition: all 0.2s ease;
}
.node-circle.completed {
    background: #10B981;
    border-color: #10B981;
    color: #FFFFFF;
}
.node-circle.active {
    background: #2874F0;
    border-color: #2874F0;
    color: #FFFFFF;
    box-shadow: 0 0 0 4px rgba(40, 116, 240, 0.2);
}
.node-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
}
.node-label.active {
    color: #2874F0;
    font-weight: 700;
}
.node-label.completed {
    color: #065F46;
    font-weight: 600;
}
.order-details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.85rem;
    background: #F8FAFC;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #EDF2F7;
    margin-top: 1.2rem;
}
.detail-item strong {
    color: #1E293B;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: block;
    margin-bottom: 0.2rem;
}
.detail-item span {
    font-size: 0.92rem;
    color: #334155;
    font-weight: 500;
}

/* History Timeline Styling */
.history-turn-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    border-left: 4px solid #2874F0;
}
.history-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #64748B;
}
.intent-badge {
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}
.intent-product_search { background: #EEF2FF; color: #4338CA; }
.intent-order_tracking { background: #FEF3C7; color: #92400E; }
.intent-product_comparison { background: #FCE7F3; color: #9D174D; }
.intent-greeting { background: #DEF7EC; color: #03543F; }
.intent-general_question { background: #F1F5F9; color: #334155; }
.intent-follow_up { background: #E0E7FF; color: #3730A3; }

/* Status Badges */
.api-badge-live {
    background-color: #DEF7EC;
    color: #03543F;
    padding: 0.25rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
}
.api-badge-fallback {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 0.25rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
}
</style>
"""
st.markdown(AMAZON_FLIPKART_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_context" not in st.session_state:
    st.session_state.chat_context = {
        "active_budget": None,
        "active_category": None,
        "last_products": [],
        "active_order_id": None,
    }

if "shopping_cart" not in st.session_state:
    st.session_state.shopping_cart = []

if "active_tab_index" not in st.session_state:
    st.session_state.active_tab_index = 0

# Initialize Memory and Core Services
memory = ConversationMemory(session_state=st.session_state)
search_engine = get_search_engine()
assistant = ShoppingAssistant(memory=memory, search_engine=search_engine, order_client=default_order_client)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def calculate_pricing_metrics(price: float) -> Dict[str, Any]:
    """Calculate realistic e-commerce M.R.P. and savings for authentic Amazon/Flipkart styling."""
    multiplier = 1.25 if price < 5000 else 1.20
    mrp = round(price * multiplier, -1)
    if mrp <= price:
        mrp = price + 500
    savings = mrp - price
    discount_pct = int(round((savings / mrp) * 100))
    return {
        "mrp": mrp,
        "savings": savings,
        "discount_pct": discount_pct,
    }


def add_product_to_cart(product: Dict[str, Any]) -> None:
    """Add a product to the user's interactive shopping cart."""
    item = {
        "id": product.get("id", "PROD"),
        "name": product.get("name", "Product"),
        "price": float(product.get("price", 0)),
        "brand": product.get("brand", "Generic"),
        "category": product.get("category", "General"),
    }
    st.session_state.shopping_cart.append(item)
    st.toast(f"✅ Added {item['name'][:32]}... to cart!", icon="🛒")


# -----------------------------------------------------------------------------
# UI Component Renderers
# -----------------------------------------------------------------------------
def render_product_card(product: Dict[str, Any], col_context: Optional[str] = "chat") -> None:
    """Render a clean, professional Amazon & Flipkart product card."""
    price = float(product.get("price", 0))
    rating = product.get("rating", "4.2")
    category = str(product.get("category", "General")).title()
    brand = product.get("brand", "Generic")
    name = product.get("name", "Product")
    desc = product.get("description", "")
    prod_id = product.get("id", "PROD")
    url = product.get("url", "#")

    pricing = calculate_pricing_metrics(price)
    mrp = pricing["mrp"]
    discount_pct = pricing["discount_pct"]

    # Category icon mapping
    category_icons = {
        "Audio": "🎧",
        "Accessories": "💻",
        "Electronics": "📱",
        "Wearables": "⌚",
        "Home Products": "🏠",
        "Cameras": "📷",
    }
    icon = category_icons.get(category, "🛍️")

    st.markdown(
        f"""
        <div class="amz-card">
            <div>
                <div class="card-header-row">
                    <span class="badge-category">{icon} {category}</span>
                    <span class="badge-assured">⚡ Assured</span>
                </div>
                <div class="product-title">{name}</div>
                <div class="product-brand">by <strong>{brand}</strong> • Item #{prod_id}</div>
                <div class="rating-row">
                    <span class="rating-pill">★ {rating}</span>
                    <span class="reviews-count">1,420+ reviews</span>
                    <span class="bought-badge">🔥 1K+ bought in past month</span>
                </div>
                <div class="product-desc">{desc}</div>
            </div>
            <div>
                <div class="pricing-block">
                    <span class="discount-tag">-{discount_pct}%</span>
                    <span class="current-price">₹{price:,.2f}</span>
                    <span class="mrp-price">M.R.P.: ₹{mrp:,.2f}</span>
                </div>
                <div class="delivery-promise">
                    <strong>FREE Delivery Tomorrow</strong> by 11 AM • Fulfilled by ShopMate
                </div>
                <div class="stock-tag">🟢 In Stock (Ships in 24 hours)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Action Buttons
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        if st.button(f"🛒 Add to Cart", key=f"cart_{col_context}_{prod_id}", use_container_width=True):
            add_product_to_cart(product)
            st.rerun()
    with c_btn2:
        if st.button(f"🔍 Compare Item", key=f"comp_{col_context}_{prod_id}", use_container_width=True):
            st.session_state["pending_input"] = f"Tell me more about {name} and how it compares"
            st.rerun()


def render_order_tracking_stepper(order: Dict[str, Any]) -> None:
    """Render an authentic Amazon & Flipkart 5-stage shipment progress tracker."""
    status = order.get("order_status", "Processing")
    order_id = order.get("order_id", "N/A")
    product_name = order.get("product_name", "N/A")
    order_date = order.get("order_date", "N/A")
    delivery_date = order.get("expected_delivery_date", "N/A")
    tracking_no = order.get("tracking_number", "N/A")
    shipping = order.get("shipping_info", {})
    carrier = shipping.get("carrier", "BlueDart Express")
    location = shipping.get("current_location", "In Transit")
    destination = shipping.get("destination", "Customer Address")

    # Stepper logic
    steps = ["Ordered", "Packed", "Shipped", "Out for Delivery", "Delivered"]
    step_weights = {
        "Processing": (1, 15),
        "Confirmed": (1, 20),
        "Packed": (2, 40),
        "Shipped": (3, 65),
        "Out for delivery": (4, 85),
        "Delivered": (5, 100),
    }

    active_step_idx, progress_pct = step_weights.get(status, (1, 20))

    nodes_html = ""
    for idx, step_name in enumerate(steps, start=1):
        if idx < active_step_idx or (status == "Delivered" and idx == 5):
            circle_cls = "node-circle completed"
            label_cls = "node-label completed"
            icon_content = "✓"
        elif idx == active_step_idx:
            circle_cls = "node-circle active"
            label_cls = "node-label active"
            icon_content = "●"
        else:
            circle_cls = "node-circle"
            label_cls = "node-label"
            icon_content = str(idx)

        nodes_html += f"""
        <div class="stepper-node">
            <div class="{circle_cls}">{icon_content}</div>
            <div class="{label_cls}">{step_name}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="amz-order-card">
            <div class="order-header-strip">
                <div>
                    <div class="order-id-label">📦 Package #{order_id}</div>
                    <div style="font-size: 0.88rem; color: #475569;">Placed on {order_date} • {product_name}</div>
                </div>
                <div>
                    <span class="assured-pill" style="font-size: 0.85rem; padding: 0.35rem 0.8rem;">
                        ● Status: {status}
                    </span>
                </div>
            </div>

            <div class="stepper-container">
                <div class="stepper-progress-track">
                    <div class="stepper-progress-fill" style="width: {progress_pct}%;"></div>
                </div>
                <div class="stepper-nodes">
                    {nodes_html}
                </div>
            </div>

            <div class="order-details-grid">
                <div class="detail-item">
                    <strong>Expected Delivery</strong>
                    <span>📅 {delivery_date}</span>
                </div>
                <div class="detail-item">
                    <strong>Logistics Partner</strong>
                    <span>🚚 {carrier}</span>
                </div>
                <div class="detail-item">
                    <strong>Waybill / Tracking</strong>
                    <span><code>{tracking_no}</code></span>
                </div>
                <div class="detail-item">
                    <strong>Live Location</strong>
                    <span>📍 {location}</span>
                </div>
                <div class="detail-item" style="grid-column: 1 / -1;">
                    <strong>Destination Address</strong>
                    <span>🏠 {destination}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# TOP NAVIGATION & BRAND BAR
# -----------------------------------------------------------------------------
cart_count = len(st.session_state.shopping_cart)
api_online = default_order_client.is_api_reachable()

st.markdown(
    f"""
    <div class="ecommerce-topbar">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
            <div>
                <span class="brand-title">🛍️ ShopMate <span class="brand-tag">in</span></span>
                <span class="assured-pill">⚡ 100% Genuine</span>
                <span style="font-size: 0.82rem; color: #E2E8F0; margin-left: 0.75rem;">
                    📍 Deliver to: <strong>Mumbai 400001</strong>
                </span>
                <div class="topbar-subtext">
                    Smart AI Shopping Assistant • Natural Language Semantic Search • Live Order Stepper
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.9rem;">
                <span style="background: rgba(255,255,255,0.12); padding: 0.35rem 0.8rem; border-radius: 8px; font-size: 0.85rem;">
                    🛒 <strong>Cart:</strong> {cart_count} {'item' if cart_count == 1 else 'items'}
                </span>
                <span class="{'api-badge-live' if api_online else 'api-badge-fallback'}">
                    {'🟢 Fast Orders API Live' if api_online else '🟡 Offline Mock Active'}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# QUICK CATEGORY PILLS BAR (Amazon / Flipkart Style)
# -----------------------------------------------------------------------------
st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #475569; margin-bottom: 0.35rem;'>EXPLORE CATEGORIES:</div>", unsafe_allow_html=True)
cat_cols = st.columns(6)
categories_config = [
    ("🎧 Audio", "Find wireless headphones and speakers"),
    ("💻 Accessories", "Show mechanical keyboards and ergonomic mice"),
    ("📱 Electronics", "Recommend tablets and smart home networking"),
    ("⌚ Wearables", "Show best smartwatches with health tracking"),
    ("🏠 Smart Home", "Find smart lights and robot vacuum cleaners"),
    ("📷 Cameras", "Show action cameras and mirrorless gear"),
]

for idx, (label, query) in enumerate(categories_config):
    with cat_cols[idx]:
        if st.button(label, key=f"cat_ribbon_{idx}", use_container_width=True):
            st.session_state["pending_input"] = query
            st.rerun()

st.markdown("<hr style='margin: 0.75rem 0 1.25rem 0; border: none; border-bottom: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR: Assistant Configuration, History Drawer & Quick Actions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛍️ ShopMate Control Center")
    st.markdown("Your 24/7 personal shopper with semantic intelligence.")

    # Status Information
    st.markdown("---")
    st.markdown("#### ⚡ System Intelligence")
    st.markdown(f"**AI Engine:** `{assistant.engine_name}`")
    if api_online:
        st.markdown("**Order Service:** <span class='api-badge-live'>FastAPI 8000 Live</span>", unsafe_allow_html=True)
    else:
        st.markdown("**Order Service:** <span class='api-badge-fallback'>Local Mock Fallback</span>", unsafe_allow_html=True)

    # Session Statistics
    if hasattr(memory, "get_session_stats"):
        stats = memory.get_session_stats()
    else:
        stats = {
            "total_messages": len(memory.messages),
            "total_turns": len(memory.messages) // 2,
            "products_recommended_count": 0,
            "unique_categories": [],
            "orders_tracked": [],
            "active_budget": memory.get_active_budget(),
            "active_category": memory.get_active_category(),
        }
    st.markdown("---")
    st.markdown("#### 📊 Session Activity")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.metric("Total Turns", stats["total_turns"])
        st.metric("Cart Items", len(st.session_state.shopping_cart))
    with s_col2:
        st.metric("Products Seen", stats["products_recommended_count"])
        st.metric("Orders Tracked", len(stats["orders_tracked"]))

    if stats["active_budget"]:
        st.caption(f"💰 Active Budget Filter: **₹{stats['active_budget']:,.2f}**")
    if stats["active_category"]:
        st.caption(f"🏷️ Active Category: **{stats['active_category'].title()}**")

    # Conversation History Exports
    st.markdown("---")
    st.markdown("#### 📜 Export Conversation")
    c_exp1, c_exp2 = st.columns(2)
    json_data = memory.export_as_json() if hasattr(memory, "export_as_json") else "{}"
    md_data = memory.export_as_markdown() if hasattr(memory, "export_as_markdown") else ""
    with c_exp1:
        st.download_button(
            label="📥 JSON",
            data=json_data,
            file_name="shopmate_conversation.json",
            mime="application/json",
            use_container_width=True,
            help="Download complete raw conversation data and entities as JSON",
        )
    with c_exp2:
        st.download_button(
            label="📄 Markdown",
            data=md_data,
            file_name="shopmate_transcript.md",
            mime="text/markdown",
            use_container_width=True,
            help="Download clean formatted shopping session transcript",
        )

    # Clear Conversation
    if st.button("🗑️ Start New Session", type="secondary", use_container_width=True):
        memory.clear()
        st.session_state.shopping_cart.clear()
        st.toast("Conversation cleared and new shopping session started!", icon="✨")
        st.rerun()

    # Recommended Quick Questions
    st.markdown("---")
    st.markdown("#### 💡 Suggested Inquiries")
    sample_queries = [
        "Find wireless headphones under ₹3000",
        "Which one has the highest rating?",
        "Show me cheaper alternatives",
        "Where is my order ORD1002?",
        "Recommend a mechanical keyboard for coding",
        "Show smartwatches with GPS under ₹7000",
    ]
    for q in sample_queries:
        if st.button(q, key=f"side_q_{q}", use_container_width=True):
            st.session_state["pending_input"] = q
            st.rerun()

    # Sample Orders Reference
    with st.expander("📦 Sample Orders Reference"):
        st.markdown("""
        - **ORD1001**: boAt Rockerz 450 (*Delivered*)
        - **ORD1002**: Sony WH-1000XM4 (*Shipped*)
        - **ORD1003**: Keychron K2 V2 (*Out for delivery*)
        - **ORD1004**: Apple iPad 10th Gen (*Packed*)
        - **ORD1005**: Portronics Toad 23 (*Confirmed*)
        - **ORD1006**: JBL Flip 6 (*Processing*)
        - **ORD1007**: Samsung Galaxy Watch 6 (*Delivered*)
        """)


# -----------------------------------------------------------------------------
# MAIN INTERFACE TABS
# -----------------------------------------------------------------------------
tab_chat, tab_history, tab_catalog, tab_cart = st.tabs([
    "💬 AI Shopping Assistant",
    "📜 Conversation History & Audit",
    "🏷️ Explore Store Catalog",
    f"🛒 Cart & Tracking ({cart_count})",
])

# =============================================================================
# TAB 1: AI Shopping Assistant
# =============================================================================
with tab_chat:
    messages = memory.get_messages()

    if not messages:
        st.info(
            "👋 **Welcome to ShopMate — Your Personal AI Shopping Concierge!**\n\n"
            "Ask me anything! For example:\n"
            "- *'Find wireless noise cancelling headphones under ₹20,000'*\n"
            "- *'Recommend an ergonomic mouse and mechanical keyboard for programming'*\n"
            "- *'Where is my order ORD1002?'*\n"
            "- *'Which product has the best customer rating?'*"
        )
    else:
        for idx, msg in enumerate(messages):
            if msg.role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg.content)
                    st.caption(f"🕒 {msg.formatted_time}")
            elif msg.role == "assistant":
                with st.chat_message("assistant", avatar="🛍️"):
                    st.markdown(msg.content)
                    st.caption(f"⚡ ShopMate • {msg.formatted_time}")

                    # Render structured product cards
                    if msg.products:
                        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; margin: 0.8rem 0 0.4rem 0;'>🛒 Featured Product Recommendations:</div>", unsafe_allow_html=True)
                        cols = st.columns(min(len(msg.products), 2))
                        for p_idx, prod in enumerate(msg.products):
                            with cols[p_idx % 2]:
                                render_product_card(prod, col_context=f"msg_{idx}_{p_idx}")

                    # Render structured order stepper card
                    if msg.order:
                        st.markdown("<div style='font-size: 1.05rem; font-weight: 700; margin: 0.8rem 0 0.4rem 0;'>📦 Shipment Tracking:</div>", unsafe_allow_html=True)
                        render_order_tracking_stepper(msg.order)

    # Chat Input
    user_input = st.chat_input("Ask for products, compare specifications, or enter an Order ID...")

    if "pending_input" in st.session_state and st.session_state["pending_input"]:
        user_input = st.session_state.pop("pending_input")

    if user_input:
        with st.spinner("ShopMate is finding the best products and checking live data..."):
            result = assistant.process_message(user_input)
        st.rerun()


# =============================================================================
# TAB 2: Conversation History & Audit
# =============================================================================
with tab_history:
    st.markdown("### 📜 Session Conversation History & Turn-by-Turn Audit")
    st.markdown("Review your entire shopping journey, prompts asked, recommendations provided, and export your session.")

    turns = memory.get_history_turns() if hasattr(memory, "get_history_turns") else []

    if not turns:
        st.write("No interactions recorded in this session yet. Ask a question in the **AI Shopping Assistant** tab to get started!")
    else:
        # History Action Controls
        h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
        json_hist = memory.export_as_json() if hasattr(memory, "export_as_json") else "{}"
        md_hist = memory.export_as_markdown() if hasattr(memory, "export_as_markdown") else ""
        with h_col1:
            st.caption(f"Total interactions recorded: **{len(turns)} turns**")
        with h_col2:
            st.download_button(
                label="📥 Export Full JSON",
                data=json_hist,
                file_name="shopmate_history.json",
                mime="application/json",
                use_container_width=True,
                key="btn_hist_json",
            )
        with h_col3:
            st.download_button(
                label="📄 Download Markdown",
                data=md_hist,
                file_name="shopmate_history.md",
                mime="text/markdown",
                use_container_width=True,
                key="btn_hist_md",
            )

        st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border: none; border-bottom: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # Render Turn Cards
        for t in reversed(turns):
            turn_idx = t["turn_index"]
            intent_val = t.get("intent", "general")
            intent_label = intent_val.replace("_", " ").title()
            u_time = t.get("user_full_time") or t.get("user_time", "")

            with st.container():
                st.markdown(
                    f"""
                    <div class="history-turn-card">
                        <div class="history-meta-row">
                            <span><strong>Turn #{turn_idx}</strong> • {u_time}</span>
                            <span class="intent-badge intent-{intent_val}">⚡ {intent_label}</span>
                        </div>
                        <div style="font-size: 0.96rem; font-weight: 700; color: #1E293B; margin-bottom: 0.4rem;">
                            👤 You: "{t['user_query']}"
                        </div>
                        <div style="font-size: 0.9rem; color: #475569; line-height: 1.5; margin-bottom: 0.6rem;">
                            🛍️ ShopMate: {t['assistant_response']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if t.get("products"):
                    with st.expander(f"📦 Viewed {len(t['products'])} Recommended Items in Turn #{turn_idx}"):
                        cols = st.columns(min(len(t["products"]), 2))
                        for p_idx, prod in enumerate(t["products"]):
                            with cols[p_idx % 2]:
                                render_product_card(prod, col_context=f"hist_{turn_idx}_{p_idx}")

                if t.get("order"):
                    with st.expander(f"🚚 Order Details for #{t['order'].get('order_id')} in Turn #{turn_idx}"):
                        render_order_tracking_stepper(t["order"])


# =============================================================================
# TAB 3: Explore Store Catalog
# =============================================================================
with tab_catalog:
    st.markdown("### 🏷️ Browse Complete ShopMate Catalog")
    st.markdown("Directly explore our catalog with verified ratings, brand filters, and instant AI follow-ups.")

    all_prods = search_engine.get_all_products()
    all_categories = ["All Categories"] + sorted(list({p.get("category", "").title() for p in all_prods}))

    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        sel_cat = st.selectbox("Filter by Category", all_categories)
    with f_col2:
        max_p = st.slider("Maximum Price (₹)", min_value=500, max_value=60000, value=60000, step=1000)
    with f_col3:
        sort_by = st.selectbox("Sort By", ["Highest Rating", "Price: Low to High", "Price: High to Low"])

    # Filter catalog
    filtered = all_prods
    if sel_cat != "All Categories":
        filtered = [p for p in filtered if p.get("category", "").title() == sel_cat]
    filtered = [p for p in filtered if float(p.get("price", 0)) <= max_p]

    if sort_by == "Highest Rating":
        filtered = sorted(filtered, key=lambda x: float(x.get("rating", 0)), reverse=True)
    elif sort_by == "Price: Low to High":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)))
    elif sort_by == "Price: High to Low":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)), reverse=True)

    st.caption(f"Showing **{len(filtered)}** products matching your criteria.")

    grid_cols = st.columns(2)
    for idx, prod in enumerate(filtered):
        with grid_cols[idx % 2]:
            render_product_card(prod, col_context=f"catalog_{idx}")


# =============================================================================
# TAB 4: Cart & Live Order Tracker
# =============================================================================
with tab_cart:
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("### 🛒 Your Shopping Cart")
        cart_items = st.session_state.shopping_cart

        if not cart_items:
            st.info("Your cart is currently empty. Click **'Add to Cart'** on any product recommendation to add it here!")
        else:
            total_cart_val = sum(item["price"] for item in cart_items)
            for c_idx, item in enumerate(cart_items):
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 700; color: #0F172A;">{item['name']}</div>
                            <div style="font-size: 0.82rem; color: #64748B;">Brand: {item['brand']} • Category: {item['category']}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 800; font-size: 1.1rem; color: #0F172A;">₹{item['price']:,.2f}</div>
                            <span class="assured-pill" style="font-size: 0.68rem;">FREE Delivery</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #334155;">Subtotal ({len(cart_items)} items):</div>
                        <div style="font-size: 0.8rem; color: #10B981; font-weight: 600;">Eligible for FREE Prime Delivery</div>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A;">
                        ₹{total_cart_val:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("Proceed to Checkout 💳", type="primary", use_container_width=True):
                    st.success("🎉 Order Placed Successfully! (Demo Simulation). A confirmation notification has been sent.")
                    st.session_state.shopping_cart.clear()
                    st.rerun()
            with btn_c2:
                if st.button("Empty Cart 🗑️", type="secondary", use_container_width=True):
                    st.session_state.shopping_cart.clear()
                    st.rerun()

    with c_right:
        st.markdown("### 📦 Instant Order Tracker")
        st.markdown("Enter any Order ID to track its real-time shipping progression:")
        lookup_id = st.text_input("Order ID", placeholder="e.g. ORD1001, ORD1002...", key="cart_order_lookup")
        if st.button("Track Shipment 🔍", use_container_width=True):
            if lookup_id:
                order_res = default_order_client.get_order(lookup_id.strip())
                if order_res:
                    render_order_tracking_stepper(order_res)
                else:
                    st.error(f"Order #{lookup_id} could not be found. Please check the Order ID or test with ORD1001 - ORD1007.")
            else:
                st.warning("Please enter a valid Order ID.")
