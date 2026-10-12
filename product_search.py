"""Product catalog loader, ChromaDB vector indexing, and semantic search engine.

Uses all-MiniLM-L6-v2 embeddings to index products from data/products.csv into
a persistent ChromaDB collection. Supports semantic search, strict numeric price filtering,
category filtering, and catalog change detection.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Expected CSV columns
REQUIRED_COLUMNS = ["id", "name", "description", "category", "price", "rating", "brand", "url"]

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "products.csv")
DEFAULT_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "ecommerce_products"


class ProductCatalogError(Exception):
    """Raised when the product catalog CSV is missing, malformed, or invalid."""
    pass


class EmbeddingModelSingleton:
    """Singleton wrapper for the embedding model to avoid repeatedly reloading it."""
    _instance = None

    @classmethod
    def get_embedding_function(cls):
        if cls._instance is None:
            # Try DefaultEmbeddingFunction (which uses all-MiniLM-L6-v2 ONNX)
            try:
                cls._instance = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                # Fallback to SentenceTransformerEmbeddingFunction if needed
                cls._instance = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
        return cls._instance


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of the CSV file to detect catalog modifications."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


class ProductSearchEngine:
    """Manages CSV validation, ChromaDB indexing, and semantic product search."""

    def __init__(
        self,
        csv_path: str = DEFAULT_CSV_PATH,
        persist_directory: str = DEFAULT_CHROMA_DIR,
    ):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_fn = EmbeddingModelSingleton.get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.catalog_df: Optional[pd.DataFrame] = None
        self._initialize_index()

    def validate_csv(self) -> pd.DataFrame:
        """Load and validate the product catalog CSV file."""
        if not os.path.exists(self.csv_path):
            raise ProductCatalogError(
                f"Product catalog file not found at: '{self.csv_path}'. "
                f"Please ensure 'data/products.csv' exists."
            )

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ProductCatalogError(f"Failed to parse CSV file: {e}")

        # Check for missing required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ProductCatalogError(
                f"Malformed CSV: missing required column(s): {', '.join(missing_cols)}"
            )

        if df.empty:
            raise ProductCatalogError("Product catalog CSV is empty.")

        # Sanitize and validate data types
        try:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        except Exception as e:
            raise ProductCatalogError(f"Failed to convert numeric fields in CSV: {e}")

        # Check for invalid rows
        if df["price"].isnull().any() or (df["price"] <= 0).any():
            raise ProductCatalogError("Invalid prices found: all products must have price > 0.")

        if df["rating"].isnull().any() or ((df["rating"] < 0) | (df["rating"] > 5)).any():
            raise ProductCatalogError("Invalid ratings found: ratings must be between 0.0 and 5.0.")

        # Clean strings
        for str_col in ["id", "name", "description", "category", "brand", "url"]:
            df[str_col] = df[str_col].astype(str).str.strip()

        # Check for duplicate product IDs in CSV
        if df["id"].duplicated().any():
            duplicates = df[df["id"].duplicated()]["id"].tolist()
            raise ProductCatalogError(f"Duplicate product IDs detected in CSV: {duplicates}")

        return df

    def _initialize_index(self) -> None:
        """Index or update products in ChromaDB, avoiding duplicate records."""
        try:
            df = self.validate_csv()
            self.catalog_df = df
        except ProductCatalogError as e:
            # Store error or handle empty catalog gracefully
            print(f"Warning: {e}")
            return

        current_hash = compute_file_hash(self.csv_path)
        hash_file = os.path.join(self.persist_directory, "catalog_hash.txt")

        stored_hash = ""
        if os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()
            except Exception:
                stored_hash = ""

        existing_count = self.collection.count()

        # If hash matches and records exist, no need to re-index
        if stored_hash == current_hash and existing_count == len(df):
            return

        # Catalog is new or changed: synchronize ChromaDB collection
        ids = df["id"].tolist()
        documents = []
        metadatas = []

        for _, row in df.iterrows():
            # Rich document text for semantic embeddings
            doc_text = (
                f"Product: {row['name']}. "
                f"Category: {row['category']}. "
                f"Brand: {row['brand']}. "
                f"Price: ₹{row['price']:,.2f}. "
                f"Rating: {row['rating']} out of 5. "
                f"Description: {row['description']}"
            )
            documents.append(doc_text)
            metadatas.append({
                "id": str(row["id"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "category": str(row["category"]).lower(),
                "price": float(row["price"]),
                "rating": float(row["rating"]),
                "brand": str(row["brand"]),
                "url": str(row["url"]),
            })

        # Upsert into ChromaDB (upsert automatically handles existing/updated IDs without duplicate errors)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        # Save new catalog hash
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            with open(hash_file, "w") as f:
                f.write(current_hash)
        except Exception:
            pass

    def search_products(
        self,
        query: str,
        n_results: int = 5,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search with strict price and category filtering.

        Args:
            query: Natural language query (e.g. "wireless headphones", "speaker for studying")
            n_results: Maximum number of products to return
            max_price: Maximum allowed price in INR (strict numeric filter)
            category: Product category filter (case-insensitive)

        Returns:
            List of structured product metadata dicts with relevance scores.
        """
        if not query or not query.strip():
            # If empty query, return top rated products matching filters
            return self._fallback_filter_df(max_price=max_price, category=category, limit=n_results)

        total_in_collection = self.collection.count()
        if total_in_collection == 0:
            return []

        # Retrieve a wider candidate pool from ChromaDB to apply strict numeric filtering
        fetch_k = min(total_in_collection, max(n_results * 4, 15))

        chroma_where = None
        if category:
            clean_category = category.strip().lower()
            chroma_where = {"category": clean_category}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=chroma_where if chroma_where else None,
            )
        except Exception as e:
            # In case where-filter fails in ChromaDB, query without where and filter in Python
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=fetch_k,
                )
            except Exception:
                return self._fallback_filter_df(max_price=max_price, category=category, limit=n_results)

        # Parse and strictly filter results
        matched_products: List[Dict[str, Any]] = []
        if results and "metadatas" in results and results["metadatas"]:
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0] if "distances" in results else []

            for i, meta in enumerate(metas):
                prod_price = float(meta.get("price", 0))
                prod_category = str(meta.get("category", "")).lower()

                # Strict price constraint
                if max_price is not None and prod_price > float(max_price):
                    continue

                # Strict category constraint
                if category and category.strip().lower() != prod_category:
                    continue

                # Cosine distance to similarity score
                similarity = 1.0 - distances[i] if i < len(distances) and distances[i] is not None else 0.85

                item = dict(meta)
                item["relevance_score"] = round(max(0.0, min(1.0, similarity)), 4)
                matched_products.append(item)

                if len(matched_products) >= n_results:
                    break

        # If zero results matched the strict filter or query, try checking dataframe directly
        if not matched_products and (max_price is not None or category is not None):
            matched_products = self._fallback_filter_df(max_price=max_price, category=category, limit=n_results)

        return matched_products

    def _fallback_filter_df(
        self,
        max_price: Optional[float] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Direct DataFrame fallback when vector search yields no items matching strict constraints."""
        if self.catalog_df is None or self.catalog_df.empty:
            try:
                self.catalog_df = self.validate_csv()
            except Exception:
                return []

        df = self.catalog_df.copy()
        if category:
            df = df[df["category"].str.lower() == category.strip().lower()]
        if max_price is not None:
            df = df[df["price"] <= float(max_price)]

        # Sort by rating descending
        df = df.sort_values(by="rating", ascending=False)
        top = df.head(limit)

        results = []
        for _, row in top.iterrows():
            item = row.to_dict()
            item["relevance_score"] = 0.70
            results.append(item)
        return results

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific product ID."""
        clean_id = (product_id or "").strip().upper()
        if self.catalog_df is not None:
            match = self.catalog_df[self.catalog_df["id"].str.upper() == clean_id]
            if not match.empty:
                return match.iloc[0].to_dict()
        return None

    def list_all_categories(self) -> List[str]:
        """List distinct categories available in catalog."""
        if self.catalog_df is not None and "category" in self.catalog_df:
            return sorted(self.catalog_df["category"].unique().tolist())
        return ["audio", "accessories", "electronics", "wearables", "home products", "cameras"]


# Module-level helper for quick search
_global_engine: Optional[ProductSearchEngine] = None


def get_search_engine(
    csv_path: str = DEFAULT_CSV_PATH,
    persist_directory: str = DEFAULT_CHROMA_DIR,
) -> ProductSearchEngine:
    global _global_engine
    if _global_engine is None or _global_engine.csv_path != csv_path:
        _global_engine = ProductSearchEngine(csv_path=csv_path, persist_directory=persist_directory)
    return _global_engine


def search_products(
    query: str,
    n_results: int = 5,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Global helper matching the required signature."""
    engine = get_search_engine()
    return engine.search_products(query=query, n_results=n_results, max_price=max_price, category=category)
