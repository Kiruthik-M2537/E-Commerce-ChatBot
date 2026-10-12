"""Intent router and entity extraction for AI E-Commerce Shopping Assistant.

Supports two-tier intent routing:
1. LLM-based structured classifier (when Gemini or OpenAI is configured)
2. Robust deterministic rule-based and semantic classifier fallback

Supported Intents:
- GREETING: Conversational greetings, politeness, farewells ("hi", "how are you", "thanks", "bye")
- PRODUCT_SEARCH: Explicit shopping requests ("find headphones under ₹3000", "show keyboards")
- PRODUCT_COMPARISON: Comparing previously viewed items ("which one has the highest rating?")
- ORDER_TRACKING: Inquiries about order/shipment status ("where is my order ORD1002?")
- FOLLOW_UP: Refining or requesting alternatives ("show me cheaper alternatives", "different brand")
- GENERAL_QUESTION: General inquiries without catalog shopping intent ("what is your return policy?")
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


# Standardized Intent Names
INTENT_GREETING = "GREETING"
INTENT_PRODUCT_SEARCH = "PRODUCT_SEARCH"
INTENT_PRODUCT_COMPARISON = "PRODUCT_COMPARISON"
INTENT_ORDER_TRACKING = "ORDER_TRACKING"
INTENT_FOLLOW_UP = "FOLLOW_UP"
INTENT_GENERAL_QUESTION = "GENERAL_QUESTION"

VALID_INTENTS = {
    INTENT_GREETING,
    INTENT_PRODUCT_SEARCH,
    INTENT_PRODUCT_COMPARISON,
    INTENT_ORDER_TRACKING,
    INTENT_FOLLOW_UP,
    INTENT_GENERAL_QUESTION,
}


class IntentStr(str):
    """Case-insensitive string with alias support for backward compatibility."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return super().__eq__(other)
        s1 = self.upper().replace("-", "_")
        s2 = other.upper().replace("-", "_")
        if s1 == s2:
            return True
        if (s1 in ("GREETING", "GENERAL_QUESTION") and s2 == "GREETING_HELP") or (
            s2 in ("GREETING", "GENERAL_QUESTION") and s1 == "GREETING_HELP"
        ):
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.upper())


@dataclass
class IntentResult:
    intent: Any
    confidence: float
    order_id: Optional[str] = None
    max_price: Optional[float] = None
    category: Optional[str] = None
    comparison_metric: Optional[str] = None  # 'rating', 'price_low', 'price_high', 'specs'
    search_query: Optional[str] = None
    raw_query: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    classified_by: str = "rules"  # 'gemini' | 'openai' | 'rules'

    def __post_init__(self):
        if not isinstance(self.intent, IntentStr):
            self.intent = IntentStr(str(self.intent))


# Robust regex for matching Order IDs (e.g. ORD1001, ord-1002, ORD_1003)
ORDER_ID_PATTERN = re.compile(r"\b(ORD[-_]?\d{3,6})\b", re.IGNORECASE)

# Price constraint patterns (e.g. 'under ₹3000', 'below 2500', 'less than 5000 rs', 'budget 2000')
PRICE_PATTERNS = [
    re.compile(r"(?:under|below|less than|within|max|budget(?: of)?)\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"(?:₹|rs\.?|inr)\s*([0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?)\s*(?:or less|max|budget)?", re.IGNORECASE),
    re.compile(r"<\s*([0-9]+(?:,[0-9]+)?)", re.IGNORECASE),
]

# Known categories and domain keywords
CATEGORY_KEYWORDS = {
    "audio": ["audio", "headphone", "headphones", "earphone", "earphones", "earbuds", "speaker", "speakers", "sound", "music", "neckband", "airpods"],
    "accessories": ["accessory", "accessories", "mouse", "keyboard", "stand", "charger", "power bank", "cable", "mat", "studying accessory"],
    "electronics": ["electronic", "electronics", "tablet", "ipad", "wifi", "mesh", "router", "raspberry pi", "gadget", "computer"],
    "wearables": ["wearable", "wearables", "watch", "smartwatch", "fitness tracker", "band"],
    "home products": ["home", "lamp", "desk lamp", "bulb", "light", "vacuum", "robovac", "smart home"],
    "cameras": ["camera", "cameras", "gimbal", "sd card", "memory card", "dslr", "gopro", "action cam", "mirrorless"],
}

# Catalog product keywords that signify actual shopping items
CATALOG_NOUNS = [
    "headphone", "headphones", "earphone", "earphones", "earbuds", "speaker", "speakers",
    "soundbar", "airpods", "neckband", "mouse", "keyboard", "keyboards", "laptop stand",
    "charger", "power bank", "tablet", "ipad", "router", "mesh wifi", "smartwatch", "watch",
    "desk lamp", "lamp", "smart bulb", "bulb", "vacuum", "robovac", "camera", "cameras",
    "gimbal", "memory card", "sd card", "accessory", "accessories", "gadget", "gadgets"
]

# Explicit shopping intent verbs/phrases
SHOPPING_VERBS = [
    "find", "search", "show", "show me", "recommend", "recommend me", "suggest", "looking for",
    "look for", "need a", "need some", "want a", "want to buy", "buy", "purchase", "browse",
    "give me", "get me", "options for", "list", "display"
]

# Greetings, conversational pleasantries, compliments, gratitude, farewells
GREETING_PATTERN = re.compile(
    r"^(?:"
    r"hi|hello|hey|howdy|sup|yo|greetings|"
    r"good\s+(?:morning|afternoon|evening|day|night)|"
    r"how\s+are\s+you(?: doing)?|how\'re\s+you|how\s+do\s+you\s+do|how\s+is\s+it\s+going|how\'s\s+it\s+going|"
    r"thanks|thank\s+you(?: so much| very much)?|thx|cheers|"
    r"nice\s+to\s+meet\s+you|pleasure|"
    r"bye|goodbye|see\s+ya|see\s+you|take\s+care"
    r")[!.,? ]*$",
    re.IGNORECASE,
)

# Comparison triggers
COMPARISON_KEYWORDS = [
    "highest rating", "best rating", "top rated", "better rating", "highest rated",
    "cheapest", "cheaper", "lowest price", "most affordable",
    "most expensive", "higher price",
    "compare", "comparison", "difference between", "which one is better",
    "which of these", "which is best", "which one has", "which is better",
    "which one", "which"
]

# Follow-up triggers
FOLLOW_UP_KEYWORDS = [
    "cheaper alternative", "cheaper alternatives", "lower price", "something cheaper",
    "more options", "show more", "anything else", "what else",
    "different brand", "other brands", "another one", "cheaper one",
    "under that", "budget option"
]


class IntentRouter:
    """Classifies user queries with LLM-assisted classification and robust deterministic fallback."""

    def __init__(self, llm_client: Any = None, model_name: Optional[str] = None):
        self.llm_client = llm_client
        self.model_name = model_name

    def route(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Route user query to an intent, extracting relevant parameters and context.

        Attempts LLM structured classification first (if client is provided),
        falling back to deterministic semantic classification.
        """
        raw_text = (query or "").strip()
        lower_text = raw_text.lower()
        context = conversation_context or {}

        # 1. Quick deterministic check for clear Order IDs (e.g. ORD1002)
        order_match = ORDER_ID_PATTERN.search(raw_text)
        detected_order_id = order_match.group(1).upper().replace("-", "").replace("_", "") if order_match else None

        is_order_phrase = any(k in lower_text for k in [
            "track", "tracking", "where is my order", "order status",
            "delivery status", "package", "shipment", "parcel", "when will my order"
        ])

        if detected_order_id or (is_order_phrase and not any(c in lower_text for c in SHOPPING_VERBS)):
            final_order_id = detected_order_id or context.get("active_order_id")
            return IntentResult(
                intent=INTENT_ORDER_TRACKING,
                confidence=0.98 if detected_order_id else 0.85,
                order_id=final_order_id,
                raw_query=raw_text,
                entities={"extracted_id": detected_order_id},
                classified_by="rules",
            )

        # 2. If an LLM client is available, attempt structured LLM classification
        if self.llm_client:
            llm_res = self._classify_with_llm(raw_text, context)
            if llm_res:
                return llm_res

        # 3. Deterministic semantic classification fallback
        return self._classify_with_rules(raw_text, context)

    def _classify_with_llm(self, query: str, context: Dict[str, Any]) -> Optional[IntentResult]:
        """Use Gemini or OpenAI structured output to classify intent reliably."""
        try:
            has_prev_products = bool(context.get("last_products"))
            active_cat = context.get("active_category")
            active_budget = context.get("active_budget")

            system_prompt = (
                "You are an intent classification engine for an e-commerce assistant.\n"
                "Classify the user message into EXACTLY ONE intent:\n"
                "- GREETING: Polite greetings, pleasantries, gratitude, farewells (e.g., 'Hi', 'how are you', 'thanks', 'bye').\n"
                "- PRODUCT_SEARCH: User explicitly wants to find, browse, or buy catalog products (e.g., 'find headphones under ₹3000', 'show keyboards').\n"
                "- PRODUCT_COMPARISON: User asks to compare previously retrieved products (e.g., 'which one has highest rating', 'which is cheaper').\n"
                "- ORDER_TRACKING: User asks about order status or provides an order ID (e.g., 'where is ORD1002').\n"
                "- FOLLOW_UP: User asks for alternatives or refines the previous search (e.g., 'show cheaper alternatives', 'different brand').\n"
                "- GENERAL_QUESTION: General questions, policies, chitchat that is NOT a product search (e.g., 'what is your return policy', 'what can you do').\n\n"
                "CRITICAL RULES:\n"
                "1. 'how are you', 'hi', 'thanks', 'thank you' MUST be classified as GREETING. NEVER as PRODUCT_SEARCH.\n"
                "2. General questions or chit-chat MUST NEVER be classified as PRODUCT_SEARCH.\n"
                "3. Only classify as PRODUCT_SEARCH if the user asks for products, features, or mentions merchandise.\n"
                f"Context: has_previous_products={has_prev_products}, active_category={active_cat}, active_budget={active_budget}.\n\n"
                "Output JSON with schema:\n"
                '{"intent": string, "confidence": number, "order_id": string|null, "max_price": number|null, '
                '"category": string|null, "comparison_metric": string|null, "search_query": string|null}'
            )

            prompt = f"User message: {query}"

            # Google Gemini Client
            if hasattr(self.llm_client, "models"):
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=300,
                )
                model_to_use = self.model_name or "gemini-3.5-flash"
                resp = self.llm_client.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=config,
                )
                raw_json = resp.text if resp and resp.text else ""

            # OpenAI Client
            elif hasattr(self.llm_client, "chat"):
                resp = self.llm_client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                raw_json = resp.choices[0].message.content
            else:
                return None

            data = json.loads(raw_json)
            intent = str(data.get("intent", "")).upper()

            if intent in VALID_INTENTS:
                return IntentResult(
                    intent=intent,
                    confidence=float(data.get("confidence", 0.95)),
                    order_id=data.get("order_id"),
                    max_price=float(data["max_price"]) if data.get("max_price") is not None else None,
                    category=data.get("category"),
                    comparison_metric=data.get("comparison_metric"),
                    search_query=data.get("search_query") or self._clean_search_query(query),
                    raw_query=query,
                    classified_by="gemini" if hasattr(self.llm_client, "models") else "openai",
                )
        except Exception as e:
            # On LLM failure or timeout, safely fall back to rules
            pass

        return None

    def _classify_with_rules(self, raw_text: str, context: Dict[str, Any]) -> IntentResult:
        """Deterministic semantic rule classifier."""
        lower_text = raw_text.lower().strip()

        # 1. Greetings, pleasantries, compliments, gratitude, check-ins
        if GREETING_PATTERN.match(lower_text):
            return IntentResult(
                intent=INTENT_GREETING,
                confidence=0.98,
                raw_query=raw_text,
                classified_by="rules",
            )

        # Common conversational phrases like "how are you doing", "thanks that's helpful", "nice to meet you"
        conversational_phrases = [
            "how are you", "how're you", "how are you doing", "how do you do",
            "what's up", "how's it going", "thanks", "thank you", "that's helpful",
            "awesome thanks", "thank you so much", "have a good day", "goodbye", "bye"
        ]
        if any(phrase in lower_text for phrase in conversational_phrases) and not any(noun in lower_text for noun in CATALOG_NOUNS):
            return IntentResult(
                intent=INTENT_GREETING,
                confidence=0.95,
                raw_query=raw_text,
                classified_by="rules",
            )

        # 2. Extract Price constraint
        detected_max_price = self._extract_price(lower_text)

        # 3. Extract Category hint
        detected_category = self._extract_category(lower_text)

        has_previous_products = bool(context.get("last_products"))

        # 4. Follow-up intent (alternatives / price reduction for previous products)
        is_follow_up_phrase = any(phrase in lower_text for phrase in FOLLOW_UP_KEYWORDS)
        if is_follow_up_phrase and has_previous_products:
            last_products = context.get("last_products", [])
            calculated_budget = detected_max_price
            if not calculated_budget and last_products:
                min_previous_price = min(p.get("price", 99999) for p in last_products)
                calculated_budget = round(min_previous_price * 0.85, 2)

            return IntentResult(
                intent=INTENT_FOLLOW_UP,
                confidence=0.92,
                max_price=calculated_budget,
                category=detected_category or context.get("active_category"),
                search_query=self._clean_search_query(raw_text),
                raw_query=raw_text,
                classified_by="rules",
            )

        # 5. Comparison intent (comparing current products)
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
                intent=INTENT_PRODUCT_COMPARISON,
                confidence=0.90,
                comparison_metric=metric,
                raw_query=raw_text,
                entities={"has_previous_products": has_previous_products},
                classified_by="rules",
            )

        # 6. Verify True Shopping Intent vs General Conversation
        has_shopping_verb = any(v in lower_text for v in SHOPPING_VERBS)
        has_catalog_noun = any(n in lower_text for n in CATALOG_NOUNS)
        has_category_hint = detected_category is not None
        has_price_constraint = detected_max_price is not None

        is_true_product_search = (
            (has_shopping_verb and (has_catalog_noun or has_category_hint or has_price_constraint))
            or (has_catalog_noun and (has_price_constraint or len(lower_text.split()) <= 6))
            or (has_category_hint and (has_price_constraint or has_shopping_verb))
        )

        if is_true_product_search:
            search_query = self._clean_search_query(raw_text)
            return IntentResult(
                intent=INTENT_PRODUCT_SEARCH,
                confidence=0.90,
                max_price=detected_max_price or context.get("active_budget"),
                category=detected_category or context.get("active_category"),
                search_query=search_query,
                raw_query=raw_text,
                entities={
                    "explicit_budget": detected_max_price is not None,
                    "explicit_category": detected_category is not None,
                },
                classified_by="rules",
            )

        # 7. Check if user is asking general questions (help, return policy, what can you do, etc.)
        return IntentResult(
            intent=INTENT_GENERAL_QUESTION,
            confidence=0.85,
            raw_query=raw_text,
            classified_by="rules",
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
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text):
                    return category
        return None

    def _clean_search_query(self, text: str) -> str:
        """Strip filler words and budget phrasing to keep clean semantic search text."""
        cleaned = text
        cleaned = re.sub(
            r"^(?:please\s+)?(?:can you\s+)?(?:show me|find|recommend|search for|look for|i need|i want|i am looking for)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?:under|below|less than|within|budget of)\s*(?:₹|rs\.?|inr)?\s*[0-9]+(?:,[0-9]+)?(?:\.[0-9]+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[₹]", "", cleaned)
        return cleaned.strip() or text
