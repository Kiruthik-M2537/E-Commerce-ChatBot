"""Conversational AI Assistant Layer.

Coordinates intent classification, ChromaDB semantic product search,
order tracking via OrderClient, conversation memory, and response generation.
Supports Google Gemini, OpenAI-compatible APIs, and reliable deterministic fallbacks.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from intent_router import (
    INTENT_GREETING,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_COMPARISON,
    INTENT_ORDER_TRACKING,
    INTENT_FOLLOW_UP,
    INTENT_GENERAL_QUESTION,
    IntentResult,
    IntentRouter,
)
from memory import ConversationMemory
from order_client import OrderClient, default_order_client
from product_search import ProductSearchEngine, get_search_engine

load_dotenv()

# Fallback model list for Gemini if the default experiences spikes or is unavailable
GEMINI_MODELS_PREFERENCE = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.8-flash"]


class ShoppingAssistant:
    """End-to-end shopping assistant coordinator with separated intent detection,

    tool execution, and response synthesis.
    """

    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        search_engine: Optional[ProductSearchEngine] = None,
        order_client: Optional[OrderClient] = None,
    ):
        self.memory = memory or ConversationMemory()
        self.search_engine = search_engine or get_search_engine()
        self.order_client = order_client or default_order_client

        # API Keys & Configurations
        self.gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        self.gemini_model_name = (os.getenv("GEMINI_MODEL_NAME") or "gemini-3.5-flash").strip()

        self.openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.openai_model_name = (os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini").strip()

        self.engine_type = "deterministic"
        self._gemini_client = None
        self._openai_client = None

        # Initialize LLM Clients
        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                self.engine_type = "gemini"
            except Exception as e:
                print(f"Warning: Could not initialize Google Gemini client: {e}")
                self._gemini_client = None
        elif self.openai_api_key:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.openai_api_key}
                if self.openai_base_url:
                    kwargs["base_url"] = self.openai_base_url
                self._openai_client = OpenAI(**kwargs)
                self.engine_type = "openai"
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self._openai_client = None

        # Router can use LLM client for structured intent classification
        active_llm = self._gemini_client or self._openai_client
        active_model = self.gemini_model_name if self._gemini_client else self.openai_model_name
        self.router = IntentRouter(llm_client=active_llm, model_name=active_model)

    @property
    def is_llm_enabled(self) -> bool:
        """Return True if an external LLM (Gemini or OpenAI) is active."""
        return self._gemini_client is not None or self._openai_client is not None

    @property
    def engine_name(self) -> str:
        """User-friendly descriptor of the active AI engine."""
        if self._gemini_client is not None:
            return f"Google Gemini ({self.gemini_model_name})"
        elif self._openai_client is not None:
            return f"OpenAI ({self.openai_model_name})"
        return "Deterministic Engine (Offline Ready)"

    def process_message(self, user_query: str) -> Dict[str, Any]:
        """Orchestrate the complete request flow:

        USER MESSAGE
        → READ RELEVANT CONTEXT
        → CLASSIFY INTENT
        → VALIDATE CLASSIFICATION
        → EXECUTE ONLY REQUIRED TOOL
        → PASS VERIFIED RESULTS TO GEMINI
        → GENERATE FINAL RESPONSE
        → UPDATE CONVERSATION MEMORY
        → RETURN STRUCTURED RESULT
        """
        clean_text = (user_query or "").strip()
        if not clean_text:
            return {
                "response_text": "Please ask a question about products or provide an order ID to track.",
                "intent": INTENT_GENERAL_QUESTION,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        # 1. Read context and classify intent
        context = self.memory.context
        routing_result: IntentResult = self.router.route(
            query=clean_text,
            conversation_context=context,
        )

        # 2. Record user query in memory
        self.memory.add_user_message(clean_text)

        # 3. Execute ONLY the required tool and synthesize response
        intent = routing_result.intent

        if intent == INTENT_GREETING:
            result = self._handle_greeting(routing_result)
        elif intent == INTENT_GENERAL_QUESTION:
            result = self._handle_general_question(routing_result)
        elif intent == INTENT_ORDER_TRACKING:
            result = self._handle_order_tracking(routing_result)
        elif intent == INTENT_PRODUCT_COMPARISON:
            result = self._handle_product_comparison(routing_result)
        elif intent == INTENT_FOLLOW_UP:
            result = self._handle_follow_up(routing_result)
        else:
            result = self._handle_product_search(routing_result)

        # 4. Update memory with results from THIS turn
        self.memory.add_assistant_message(
            content=result["response_text"],
            products=result.get("products") or None,
            order=result.get("order"),
            metadata={"intent": intent, "mode": result.get("mode")},
        )

        return result

    def _handle_greeting(self, routing: IntentResult) -> Dict[str, Any]:
        """Handle greetings and pleasantries WITHOUT executing product search."""
        fallback_text = (
            "Hello! I am ShopMate, your AI shopping assistant. How can I help you today? "
            "You can ask me to find products, compare items, check prices, or track an order!"
        )

        response_text = fallback_text
        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate, a friendly, professional AI e-commerce assistant. "
                    "Respond warmly to the user's greeting or pleasantry. "
                    "Keep your response natural, friendly, and concise (1-2 sentences). "
                    "DO NOT mention or list any products unless the user asks for shopping help."
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_GREETING,
            "products": [],  # Crucial: NO product cards rendered
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_general_question(self, routing: IntentResult) -> Dict[str, Any]:
        """Handle general inquiries, capabilities, or store questions WITHOUT product retrieval."""
        fallback_text = (
            "I am your AI Shopping Assistant! I can help you search our catalog for electronics, "
            "audio products, accessories, cameras, wearables, and home appliances. "
            "I can also compare product specifications and track orders using order IDs like ORD1001."
        )

        response_text = fallback_text
        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate, a helpful e-commerce shopping assistant. "
                    "Answer the user's question politely and concisely. "
                    "Explain that you help customers find products, compare prices/ratings in Indian Rupees (₹), "
                    "and track orders. DO NOT invent specific catalog products."
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_GENERAL_QUESTION,
            "products": [],
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_order_tracking(self, routing: IntentResult) -> Dict[str, Any]:
        """Retrieve order data from OrderClient and format verified status."""
        order_id = routing.order_id

        if not order_id:
            msg = (
                "To track your order, please provide your Order ID (for example: **ORD1001**, **ORD1002**, or **ORD1003**)."
            )
            return {
                "response_text": msg,
                "intent": INTENT_ORDER_TRACKING,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        # Tool execution: Order API lookup
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
                "intent": INTENT_ORDER_TRACKING,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        order = res["order"]
        shipping = order.get("shipping_info", {})
        source_label = "Live API Service" if res["source"] == "live_api" else "Local Mock Fallback"

        deterministic_text = (
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

        response_text = deterministic_text
        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate AI assistant. Explain the order tracking status using ONLY the verified facts below.\n"
                    "CRITICAL GROUNDING: Do NOT invent delivery updates, locations, or dates.\n"
                    f"ORDER DATA: {order}\nSource: {source_label}"
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_ORDER_TRACKING,
            "products": [],
            "order": order,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_product_comparison(self, routing: IntentResult) -> Dict[str, Any]:
        """Compare previously recommended products from conversation memory."""
        previous_products = self.memory.get_last_recommended_products()

        if not previous_products:
            msg = (
                "I don't have any previously recommended products in our current conversation to compare yet.\n\n"
                "Try searching for some items first (e.g. *'Show me wireless headphones under ₹3000'*), "
                "and I'll gladly compare their ratings, prices, and features!"
            )
            return {
                "response_text": msg,
                "intent": INTENT_PRODUCT_COMPARISON,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        metric = routing.comparison_metric or "specs"

        if metric == "rating":
            best = max(previous_products, key=lambda p: float(p.get("rating", 0)))
            text = (
                f"⭐ **Highest Rated Product:**\n\n"
                f"Among the items we discussed, **{best['name']}** has the highest rating with **{best['rating']} ★**!\n\n"
                f"- **Brand:** {best['brand']}\n"
                f"- **Price:** ₹{float(best['price']):,.2f}\n"
                f"- **Category:** {best['category'].title()}\n"
                f"- **Highlight:** {best['description']}\n\n"
                f"**Compared Ratings:**\n"
            )
            for p in sorted(previous_products, key=lambda x: float(x.get("rating", 0)), reverse=True):
                text += f"- **{p['name']}**: {p['rating']} ★ (₹{float(p['price']):,.2f})\n"

        elif metric == "price_low":
            cheapest = min(previous_products, key=lambda p: float(p.get("price", 999999)))
            text = (
                f"💰 **Most Affordable Option:**\n\n"
                f"**{cheapest['name']}** is the lowest-priced option at **₹{float(cheapest['price']):,.2f}** "
                f"with a **{cheapest['rating']} ★** rating.\n\n"
                f"- **Brand:** {cheapest['brand']}\n"
                f"- **Category:** {cheapest['category'].title()}\n"
                f"- **Details:** {cheapest['description']}\n\n"
                f"**Price Comparison (Low to High):**\n"
            )
            for p in sorted(previous_products, key=lambda x: float(x.get("price", 0))):
                text += f"- **{p['name']}**: ₹{float(p['price']):,.2f} ({p['rating']} ★)\n"
        else:
            text = "🔍 **Product Comparison Overview:**\n\n"
            for i, p in enumerate(previous_products, 1):
                text += (
                    f"**{i}. {p['name']}**\n"
                    f"- Price: ₹{float(p['price']):,.2f} | Rating: {p['rating']} ★ | Brand: {p['brand']}\n"
                    f"- Highlight: {p['description']}\n\n"
                )

        response_text = text
        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate AI assistant comparing products discussed in conversation.\n"
                    "Compare them using ONLY the attributes provided below. Format prices in ₹.\n"
                    f"PRODUCTS TO COMPARE: {previous_products}\n"
                    f"User focus: {metric}"
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_PRODUCT_COMPARISON,
            "products": previous_products,
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _handle_follow_up(self, routing: IntentResult) -> Dict[str, Any]:
        """Handle follow-up queries such as cheaper alternatives or refining criteria."""
        category = routing.category or self.memory.get_active_category()
        max_price = routing.max_price

        # Update context entities
        if max_price:
            self.memory.set_context_entity("active_budget", max_price)
        if category:
            self.memory.set_context_entity("active_category", category)

        # Tool execution: search ChromaDB
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
                "intent": INTENT_FOLLOW_UP,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        price_clause = f" priced under ₹{max_price:,.2f}" if max_price else ""
        response_text = f"Here are great alternative options{price_clause} based on our discussion:"

        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate AI assistant recommending alternative products.\n"
                    "Summarize why these alternatives are great options based on the user's criteria.\n"
                    f"PRODUCTS: {products}\nCategory: {category}\nBudget: {max_price}"
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_FOLLOW_UP,
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

        # Tool execution: search ChromaDB
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
                "intent": INTENT_PRODUCT_SEARCH,
                "products": [],
                "order": None,
                "mode": "deterministic",
            }

        price_badge = f" under ₹{max_price:,.2f}" if max_price else ""
        response_text = f"I found {len(products)} relevant products{price_badge} matching your request:"

        if self.is_llm_enabled:
            llm_text = self._call_llm(
                system_prompt=(
                    "You are ShopMate, a professional AI shopping assistant.\n"
                    "Summarize these product recommendations concisely and helpfully. "
                    "Highlight prices (in ₹), ratings, and key features. "
                    "Do NOT fabricate any specs or products not listed below.\n"
                    f"RETRIEVED CATALOG PRODUCTS: {products}"
                ),
                user_message=routing.raw_query,
            )
            if llm_text:
                response_text = llm_text

        return {
            "response_text": response_text,
            "intent": INTENT_PRODUCT_SEARCH,
            "products": products,
            "order": None,
            "mode": "llm" if self.is_llm_enabled else "deterministic",
        }

    def _call_llm(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Invoke Gemini or OpenAI with error handling and fallback model recovery."""
        # 1. Try Gemini
        if self._gemini_client is not None:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=600,
            )

            # Build conversational turns
            contents = []
            for turn in self.memory.get_conversation_history_for_prompt(max_turns=3):
                role = "user" if turn["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])])
                )
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
            )

            # Candidate model list to try
            models_to_try = [self.gemini_model_name] + [
                m for m in GEMINI_MODELS_PREFERENCE if m != self.gemini_model_name
            ]

            for model_candidate in models_to_try:
                try:
                    resp = self._gemini_client.models.generate_content(
                        model=model_candidate,
                        contents=contents,
                        config=config,
                    )
                    if resp and resp.text:
                        self.gemini_model_name = model_candidate
                        return resp.text.strip()
                except Exception as e:
                    # Model not available or 503, try next candidate
                    continue

        # 2. Try OpenAI
        if self._openai_client is not None:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.memory.get_conversation_history_for_prompt(max_turns=3))
            messages.append({"role": "user", "content": user_message})

            try:
                resp = self._openai_client.chat.completions.create(
                    model=self.openai_model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        return None
