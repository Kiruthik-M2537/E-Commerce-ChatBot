"""Intent router and entity extraction for AI E-Commerce Shopping Assistant.

Classifies incoming user messages into distinct intents:
- order_tracking
- product_comparison
- product_search
- follow_up
- greeting_help
- clarification

Extracts relevant parameters such as order_id, max_price, and category hints.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IntentResult:
    intent: str
    confidence: float
    order_id: Optional[str] = None
    max_price: Optional[float] = None
    category: Optional[str] = None
    comparison_metric: Optional[str] = None  # 'rating', 'price_low', 'price_high', 'specs'
    search_query: Optional[str] = None
    raw_query: str = ""
    entities: Dict[str, Any] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}


# Standard regex for matching Order IDs (e.g. ORD1001, ord-1002, ORD_1003)
ORDER_ID_PATTERN = re.compile(r"\b(ORD[-_]?\d{3,6})\b", re.IGNORECASE)

# Price constraint patterns (e.g. 'under ₹3000', 'below 2500', 'less than 5000 rs', 'budget 2000')
PRICE_PATTERNS = [
    re.compile(r"(?:under|below|less than|within|max|budget(?: of)?)\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"(?:₹|rs\.?|inr)\s*([0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?)\s*(?:or less|max|budget)?", re.IGNORECASE),
    re.compile(r"<\s*([0-9]+(?:,[0-9]+)?)", re.IGNORECASE),
]

# Known categories in product catalog
CATEGORY_KEYWORDS = {
    "audio": ["audio", "headphone", "headphones", "earphone", "earphones", "earbuds", "speaker", "speakers", "sound", "music", "neckband", "airpods"],
    "accessories": ["accessory", "accessories", "mouse", "keyboard", "stand", "charger", "power bank", "cable", "mat", "studying accessory"],
    "electronics": ["electronic", "electronics", "tablet", "ipad", "wifi", "mesh", "router", "raspberry pi", "gadget", "computer"],
    "wearables": ["wearable", "wearables", "watch", "smartwatch", "fitness tracker", "band"],
    "home products": ["home", "lamp", "desk lamp", "bulb", "light", "vacuum", "robovac", "smart home"],
    "cameras": ["camera", "cameras", "gimbal", "sd card", "memory card", "dslr", "gopro", "action cam", "mirrorless"],
}

# Comparison trigger words
COMPARISON_KEYWORDS = [
    "highest rating", "best rating", "top rated", "better rating",
    "cheapest", "cheaper", "lowest price", "most affordable",
    "most expensive", "higher price",
    "compare", "comparison", "difference between", "which one is better",
    "which of these", "which is best", "which one has",
]

# Follow-up triggers
FOLLOW_UP_KEYWORDS = [
    "cheaper alternative", "cheaper alternatives", "lower price",
    "more options", "show more", "anything else", "what else",
    "under that", "different brand", "other brands",
]

# Greetings and help
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "help", "what can you do", "how does this work", "who are you", "features",
]


class IntentRouter:
    """Classifies user queries and extracts e-commerce entities."""

    def route(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Route user query to an intent, extracting relevant parameters and context."""
        raw_text = (query or "").strip()
        lower_text = raw_text.lower()
        context = conversation_context or {}

        # 1. Extract Order ID
        order_match = ORDER_ID_PATTERN.search(raw_text)
        detected_order_id = order_match.group(1).upper().replace("-", "").replace("_", "") if order_match else None

        # 2. Extract Price constraint
        detected_max_price = self._extract_price(lower_text)

        # 3. Extract Category hint
        detected_category = self._extract_category(lower_text)

        # If order ID is present or tracking phrases without products are used
        is_order_query = any(k in lower_text for k in [
            "track", "tracking", "where is my order", "order status",
            "delivery status", "package", "shipment", "parcel", "when will my order"
        ])

        if detected_order_id or (is_order_query and not any(c in lower_text for c in ["buy", "search", "recommend", "show me"])):
            # If user said "track my order" without ID, check if there's an active_order_id in context
            final_order_id = detected_order_id or context.get("active_order_id")
            return IntentResult(
                intent="order_tracking",
                confidence=0.95 if detected_order_id else 0.80,
                order_id=final_order_id,
                raw_query=raw_text,
                entities={"extracted_id": detected_order_id},
            )

        # 4. Check for Follow-up Intent (e.g. "show me cheaper alternatives", "show more")
        has_previous_products = bool(context.get("last_products"))
        is_follow_up_phrase = any(phrase in lower_text for phrase in FOLLOW_UP_KEYWORDS)
        if is_follow_up_phrase and has_previous_products:
            # If user said "show cheaper alternatives", lower the max_price or use the lowest of previous products
            last_products = context.get("last_products", [])
            calculated_budget = detected_max_price
            if not calculated_budget and last_products:
                min_previous_price = min(p.get("price", 99999) for p in last_products)
                calculated_budget = round(min_previous_price * 0.85, 2)

            return IntentResult(
                intent="follow_up",
                confidence=0.92,
                max_price=calculated_budget,
                category=detected_category or context.get("active_category"),
                search_query=self._clean_search_query(raw_text),
                raw_query=raw_text,
            )

        # 5. Check for Comparison Intent (e.g. "which one has highest rating", "which is cheaper")
        is_comparison_phrase = any(phrase in lower_text for phrase in COMPARISON_KEYWORDS)
        if is_comparison_phrase and (has_previous_products or "which" in lower_text or "compare" in lower_text):
            metric = "specs"
            if any(k in lower_text for k in ["highest rating", "best rating", "top rated", "better rating", "highest rated"]):
                metric = "rating"
            elif any(k in lower_text for k in ["cheapest", "cheaper", "lowest price", "most affordable"]):
                metric = "price_low"
            elif any(k in lower_text for k in ["most expensive", "higher price"]):
                metric = "price_high"

            return IntentResult(
                intent="product_comparison",
                confidence=0.90,
                comparison_metric=metric,
                raw_query=raw_text,
                entities={"has_previous_products": has_previous_products},
            )

        # 6. Check for Greetings / Help
        # Check if the query is strictly or predominantly a greeting
        words = re.findall(r"\b\w+\b", lower_text)
        if len(words) <= 4 and any(g in lower_text for g in GREETING_KEYWORDS) and not detected_category and not detected_max_price:
            return IntentResult(
                intent="greeting_help",
                confidence=0.95,
                raw_query=raw_text,
            )

        # 7. Default to Product Search
        search_query = self._clean_search_query(raw_text)
        return IntentResult(
            intent="product_search",
            confidence=0.85,
            max_price=detected_max_price or context.get("active_budget"),
            category=detected_category or context.get("active_category"),
            search_query=search_query,
            raw_query=raw_text,
            entities={
                "explicit_budget": detected_max_price is not None,
                "explicit_category": detected_category is not None,
            },
        )

    def _extract_price(self, text: str) -> Optional[float]:
        """Parse numeric budget constraints from natural language text."""
        for pattern in PRICE_PATTERNS:
            match = pattern.search(text)
            if match:
                price_str = match.group(1).replace(",", "")
                try:
                    val = float(price_str)
                    if 10 <= val <= 1000000:
                        return val
                except ValueError:
                    continue
        return None

    def _extract_category(self, text: str) -> Optional[str]:
        """Detect product category from query text."""
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                # Match full word or multi-word phrase
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text):
                    return category
        return None

    def _clean_search_query(self, text: str) -> str:
        """Strip filler words and budget phrasing to keep clean semantic search text."""
        cleaned = text
        # Remove common request prefixes
        cleaned = re.sub(r"^(?:please\s+)?(?:can you\s+)?(?:show me|find|recommend|search for|look for|i need|i want|i am looking for)\s+", "", cleaned, flags=re.IGNORECASE)
        # Remove budget suffix phrases like "under 3000", "below ₹2500"
        cleaned = re.sub(r"(?:under|below|less than|within|budget of)\s*(?:₹|rs\.?|inr)?\s*[0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?", "", cleaned, flags=re.IGNORECASE)
        # Remove remaining currency symbols
        cleaned = re.sub(r"[₹]", "", cleaned)
        return cleaned.strip() or text
