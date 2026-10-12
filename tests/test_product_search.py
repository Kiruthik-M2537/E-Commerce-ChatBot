"""Tests for product catalog validation, ChromaDB indexing, and semantic search."""

import os
import tempfile
import pytest
import pandas as pd

from product_search import (
    ProductSearchEngine,
    ProductCatalogError,
    DEFAULT_CSV_PATH,
)


@pytest.fixture(scope="module")
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing with Windows-safe cleanup."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def search_engine(temp_chroma_dir):
    """Instantiate a ProductSearchEngine using standard products.csv and temporary ChromaDB."""
    return ProductSearchEngine(csv_path=DEFAULT_CSV_PATH, persist_directory=temp_chroma_dir)


def test_csv_validation_success(search_engine):
    """Verify that default products.csv passes validation and has at least 20 items."""
    df = search_engine.validate_csv()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 20
    assert "price" in df.columns
    assert "category" in df.columns
    assert (df["price"] > 0).all()
    assert (df["rating"] >= 0).all()
    assert (df["rating"] <= 5).all()


def test_missing_csv_raises_error(temp_chroma_dir):
    """Verify appropriate error when CSV path does not exist."""
    with pytest.raises(ProductCatalogError) as exc_info:
        engine = ProductSearchEngine(csv_path="non_existent_file.csv", persist_directory=temp_chroma_dir)
        engine.validate_csv()
    assert "not found" in str(exc_info.value).lower()


def test_malformed_csv_missing_columns(temp_chroma_dir):
    """Verify error when required columns are missing."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,name,price\n1,Test Item,100\n")
        bad_csv_path = f.name

    try:
        with pytest.raises(ProductCatalogError) as exc_info:
            engine = ProductSearchEngine(csv_path=bad_csv_path, persist_directory=temp_chroma_dir)
            engine.validate_csv()
        assert "missing required column" in str(exc_info.value).lower()
    finally:
        if os.path.exists(bad_csv_path):
            os.remove(bad_csv_path)


def test_semantic_search_retrieves_headphones(search_engine):
    """Verify semantic retrieval finds audio products for headphone query."""
    results = search_engine.search_products(query="wireless noise cancelling headphones", n_results=3)
    assert len(results) > 0
    top_result = results[0]
    assert "name" in top_result
    assert "price" in top_result
    # Should recommend headphone or audio product at top
    assert any(term in top_result["name"].lower() for term in ["headphone", "audio", "sony", "boat", "airpods"])


def test_strict_numeric_price_filtering(search_engine):
    """Verify that max_price strictly enforces price limits with zero violations."""
    max_allowed = 2000.0
    results = search_engine.search_products(
        query="audio headphones earphones",
        n_results=5,
        max_price=max_allowed,
    )
    assert len(results) > 0
    for prod in results:
        assert float(prod["price"]) <= max_allowed, f"Product {prod['name']} price {prod['price']} exceeds {max_allowed}"


def test_category_filtering(search_engine):
    """Verify category filtering strictly returns items matching category."""
    results = search_engine.search_products(
        query="gadget",
        n_results=5,
        category="cameras",
    )
    assert len(results) > 0
    for prod in results:
        assert prod["category"].lower() == "cameras"


def test_empty_results_when_budget_too_low(search_engine):
    """Verify handling when no product meets an impossibly low budget constraint."""
    results = search_engine.search_products(
        query="laptop accessories",
        n_results=5,
        max_price=50.0,  # No accessory is under 50 rupees
    )
    assert len(results) == 0
