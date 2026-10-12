"""Conversational AI Assistant Layer.

Coordinates intent classification, ChromaDB semantic product search,
order tracking via OrderClient, conversation memory, and response generation
supporting both LLM-powered (OpenAI-compatible) and deterministic fallback modes.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from intent_router import IntentResult, IntentRouter
from memory import ConversationMemory
from order_client import OrderClient, default_order_client
from product_search import ProductSearchEngine, get_search_engine

load_dotenv()


class ShoppingAssistant:
    """End-to-end shopping assistant coordinator."""

    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        search_engine: Optional[ProductSearchEngine] = None,
        order_client: Optional[OrderClient] = None,
    ):
        self.memory = memory or ConversationMemory()
        self.search_engine = search_engine or get_search_engine()
        self.order_client = order_client or default_order_client
        self.router = IntentRouter()

        # Optional OpenAI configuration
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip()

        self._llm_client = None
        if self.api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._llm_client = OpenAI(**kwargs)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}. Falling back to deterministic mode.")
                self._llm_client = None

    @property
    def is_llm_enabled(self) -> bool:
        """Return True if an external LLM is configured and ready."""
        return self._llm_client is not None

    def process_message(self, user_query: str) -> Dict[str, Any]:
        """Process incoming user message through routing, retrieval, and generation."""
        clean_text = (user_query or "").strip()
        if not clean_text:
            return {
                "response_text": "Please ask a question about products or provide an order ID to track.",
                "intent": "clarification",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        # 1. Intent routing
        routing_result: IntentResult = self.router.route(
            query=clean_text,
            conversation_context=self.memory.context,
        )

        # Record user query in memory
        self.memory.add_user_message(clean_text)

        # 2. Dispatch to intent handler
        if routing_result.intent == "order_tracking":
            result = self._handle_order_tracking(routing_result)
        elif routing_result.intent == "product_comparison":
            result = self._handle_product_comparison(routing_result)
        elif routing_result.intent == "follow_up":
            result = self._handle_follow_up(routing_result)
        elif routing_result.intent == "greeting_help":
            result = self._handle_greeting_help(routing_result)
        else:
            result = self._handle_product_search(routing_result)

        # 3. Record assistant response in memory
        self.memory.add_assistant_message(
            content=result["response_text"],
            products=result.get("products"),
            order=result.get("order"),
            metadata={"intent": result.get("intent"), "mode": result.get("mode")},
        )

        return result

    def _handle_order_tracking(self, routing: IntentResult) -> Dict[str, Any]:
        """Handle order tracking queries."""
        order_id = routing.order_id

        if not order_id:
            msg = (
                "To track your order, please provide your order ID (for example: **ORD1001**, **ORD1002**, or **ORD1003**)."
            )
            return {
                "response_text": msg,
                "intent": "order_tracking",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        # Query order client (live API with fallback)
        res = self.order_client.get_order(order_id)

        if not res["success"]:
            msg = (
                f"❌ **Order Not Found**: {res['error']}\n\n"
                f"Please check your order number. Try one of our demo orders:\n"
                f"- **ORD1001** (Delivered)\n"
                f"- **ORD1002** (Shipped / In Transit)\n"
                f"- **ORD1003** (Out for delivery)\n"
                f"- **ORD1004** (Packed)\n"
                f"- **ORD1005** (Confirmed)\n"
                f"- **ORD1006** (Processing)\n"
                f"- **ORD1007** (Delivered)"
            )
            return {
                "response_text": msg,
                "intent": "order_tracking",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        order = res["order"]
        shipping = order.get("shipping_info", {})
        source_label = "Live API Service" if res["source"] == "live_api" else "Local Mock Fallback"

        # Deterministic formatting
        response_text = (
            f"📦 **Order Status for #{order['order_id']}**\n\n"
            f"- **Product:** {order['product_name']}\n"
            f"- **Status:** **{order['order_status']}**\n"
            f"- **Order Date:** {order['order_date']}\n"
            f"- **Estimated Delivery:** {order['expected_delivery_date']}\n"
            f"- **Carrier:** {shipping.get('carrier', 'Standard Delivery')}\n"
            f"- **Tracking Number:** `{order['tracking_number']}`\n"
            f"- **Destination:** {shipping.get('destination', 'Customer Address')}\n"
            f"- **Current Location:** {shipping.get('current_location', 'In Transit')}\n\n"
            f"*(Source: {source_label})*"
        )

        # If LLM enabled, we can let LLM summarize grounded strictly on this order
        if self.is_llm_enabled:
            llm_text = self._generate_llm_response(
                user_query=routing.raw_query,
                context_type="order",
                data_summary=f"ORDER DATA: {order}. Source: {source_label}. DO NOT INVENT TRACKING UPDATES.",
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": "order_tracking",
            "products": [],
            "order": order,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_product_comparison(self, routing: IntentResult) -> Dict[str, Any]:
        """Compare previously recommended products."""
        previous_products = self.memory.get_last_recommended_products()

        if not previous_products:
            msg = (
                "I don't have any previously recommended products in this conversation to compare yet.\n\n"
                "Try searching for some items first (e.g. *'Show me wireless headphones under ₹3000'*), "
                "and I'll gladly compare their ratings, prices, and features for you!"
            )
            return {
                "response_text": msg,
                "intent": "product_comparison",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        metric = routing.comparison_metric or "specs"

        if metric == "rating":
            best = max(previous_products, key=lambda p: float(p.get("rating", 0)))
            text = (
                f"⭐ **Highest Rated Product:**\n\n"
                f"Among the items we discussed, **{best['name']}** has the highest customer rating with **{best['rating']} ★**!\n\n"
                f"- **Brand:** {best['brand']}\n"
                f"- **Price:** ₹{float(best['price']):,.2f}\n"
                f"- **Category:** {best['category'].title()}\n"
                f"- **Key Highlight:** {best['description']}\n\n"
                f"**All Compared Ratings:**\n"
            )
            for p in sorted(previous_products, key=lambda x: float(x.get("rating", 0)), reverse=True):
                text += f"- **{p['name']}**: {p['rating']} ★ (₹{float(p['price']):,.2f})\n"

        elif metric == "price_low":
            cheapest = min(previous_products, key=lambda p: float(p.get("price", 999999)))
            text = (
                f"💰 **Most Budget-Friendly Option:**\n\n"
                f"**{cheapest['name']}** is the lowest-priced option at **₹{float(cheapest['price']):,.2f}** "
                f"with a **{cheapest['rating']} ★** rating.\n\n"
                f"- **Brand:** {cheapest['brand']}\n"
                f"- **Category:** {cheapest['category'].title()}\n"
                f"- **Details:** {cheapest['description']}\n\n"
                f"**Price Comparison (Low to High):**\n"
            )
            for p in sorted(previous_products, key=lambda x: float(x.get("price", 0))):
                text += f"- **{p['name']}**: ₹{float(p['price']):,.2f} ({p['rating']} ★)\n"

        elif metric == "price_high":
            premium = max(previous_products, key=lambda p: float(p.get("price", 0)))
            text = (
                f"💎 **Top Premium Product:**\n\n"
                f"**{premium['name']}** is the top-tier selection at **₹{float(premium['price']):,.2f}** "
                f"rated **{premium['rating']} ★**.\n\n"
                f"- **Brand:** {premium['brand']}\n"
                f"- **Details:** {premium['description']}\n"
            )

        else:
            text = "🔍 **Product Comparison Overview:**\n\n"
            for i, p in enumerate(previous_products, 1):
                text += (
                    f"**{i}. {p['name']}**\n"
                    f"- Price: ₹{float(p['price']):,.2f} | Rating: {p['rating']} ★ | Brand: {p['brand']}\n"
                    f"- Highlight: {p['description']}\n\n"
                )

        if self.is_llm_enabled:
            llm_text = self._generate_llm_response(
                user_query=routing.raw_query,
                context_type="comparison",
                data_summary=f"PRODUCTS TO COMPARE: {previous_products}. User focus: {metric}.",
            )
            if llm_text:
                text = llm_text

        return {
            "response_text": text,
            "intent": "product_comparison",
            "products": previous_products,
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_follow_up(self, routing: IntentResult) -> Dict[str, Any]:
        """Handle follow-up queries such as cheaper alternatives or refining criteria."""
        category = routing.category or self.memory.get_active_category()
        max_price = routing.max_price

        # Update memory context entities
        if max_price:
            self.memory.set_context_entity("active_budget", max_price)
        if category:
            self.memory.set_context_entity("active_category", category)

        products = self.search_engine.search_products(
            query=routing.search_query or category or "affordable",
            n_results=5,
            max_price=max_price,
            category=category,
        )

        if not products:
            price_str = f" under ₹{max_price:,.2f}" if max_price else ""
            cat_str = f" in {category}" if category else ""
            text = (
                f"I couldn't find any alternative products{cat_str}{price_str} in our catalog. "
                f"Would you like to explore another category or increase the budget slightly?"
            )
            return {
                "response_text": text,
                "intent": "follow_up",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        price_clause = f" priced under ₹{max_price:,.2f}" if max_price else ""
        text = f"Here are great alternative options{price_clause} based on our discussion:"

        if self.is_llm_enabled:
            llm_text = self._generate_llm_response(
                user_query=routing.raw_query,
                context_type="products",
                data_summary=f"MATCHING PRODUCTS: {products}. Category: {category}. Max price: {max_price}.",
            )
            if llm_text:
                text = llm_text

        return {
            "response_text": text,
            "intent": "follow_up",
            "products": products,
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_product_search(self, routing: IntentResult) -> Dict[str, Any]:
        """Execute semantic search with strict numeric price and category constraints."""
        max_price = routing.max_price
        category = routing.category

        # Store persistent context entities
        if max_price:
            self.memory.set_context_entity("active_budget", max_price)
        if category:
            self.memory.set_context_entity("active_category", category)

        products = self.search_engine.search_products(
            query=routing.search_query or routing.raw_query,
            n_results=5,
            max_price=max_price,
            category=category,
        )

        if not products:
            price_msg = f" under ₹{max_price:,.2f}" if max_price else ""
            cat_msg = f" in category '{category}'" if category else ""
            categories_available = ", ".join(self.search_engine.list_all_categories())
            text = (
                f"No products found matching '{routing.raw_query}'{price_msg}{cat_msg}.\n\n"
                f"💡 **Suggestions:**\n"
                f"- Try adjusting or increasing the budget limit.\n"
                f"- Browse our available categories: **{categories_available}**.\n"
                f"- Search for general terms like *'wireless headphones'*, *'desk lamp'*, or *'smartwatch'*."
            )
            return {
                "response_text": text,
                "intent": "product_search",
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        price_badge = f" under ₹{max_price:,.2f}" if max_price else ""
        text = f"I found {len(products)} relevant products{price_badge} matching your request:"

        if self.is_llm_enabled:
            llm_text = self._generate_llm_response(
                user_query=routing.raw_query,
                context_type="products",
                data_summary=f"MATCHING PRODUCTS: {products}. Max price constraint: {max_price}.",
            )
            if llm_text:
                text = llm_text

        return {
            "response_text": text,
            "intent": "product_search",
            "products": products,
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_greeting_help(self, routing: IntentResult) -> Dict[str, Any]:
        """Provide friendly onboarding and capabilities overview."""
        text = (
            "👋 **Welcome to ShopMate — Your AI Shopping Assistant!**\n\n"
            "I can assist you with:\n"
            "1. 🔍 **Semantic Product Search:** *'Find wireless headphones under ₹3000'* or *'Show laptop study accessories'*\n"
            "2. ⚖️ **Product Comparisons:** *'Which of those has the highest rating?'* or *'Show cheaper alternatives'*\n"
            "3. 📦 **Live Order Tracking:** *'Where is my order ORD1002?'* or *'Track order ORD1003'*\n"
            "4. 💡 **Budget Recommendations:** Filter by price and categories accurately.\n\n"
            "What would you like to explore today?"
        )
        return {
            "response_text": text,
            "intent": "greeting_help",
            "products": [],
            "order": None,
            "mode": "deterministic",
        }

    def _generate_llm_response(
        self,
        user_query: str,
        context_type: str,
        data_summary: str,
    ) -> Optional[str]:
        """Invoke external OpenAI-compatible model with strict grounding rules."""
        if not self._llm_client:
            return None

        system_prompt = (
            "You are a professional, helpful, and concise AI E-Commerce Shopping Assistant. "
            "CRITICAL GROUNDING RULES:\n"
            "1. NEVER invent products, prices, ratings, features, or order details not present in the provided context.\n"
            "2. For order tracking, ONLY state facts confirmed by the order data. Never invent shipping locations or delivery promises.\n"
            "3. Always format prices in Indian Rupees (₹).\n"
            "4. Keep responses crisp, polite, and actionable for shoppers.\n"
            f"CONTEXT DATA:\n{data_summary}"
        )

        # Retrieve recent message turns for conversational continuity
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_conversation_history_for_prompt(max_turns=3))
        messages.append({"role": "user", "content": user_query})

        try:
            response = self._llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM generation failed: {e}. Falling back to deterministic response.")
            return None
