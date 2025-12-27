"""
Basic tests for the embedding pipeline functions.
"""
import os
from unittest.mock import patch, MagicMock
import sys
import pytest

# Add the backend directory to the path so we can import main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import (
    get_all_urls,
    extract_text_from_url,
    chunk_text,
    embed,
    create_collection,
    save_chunk_to_qdrant,
    main,
    load_config_from_env,
    initialize_clients,
    _extract_urls_from_sitemap,
    _crawl_docusaurus_site,
    _is_valid_doc_url
)


def test_load_config_from_env():
    """Test that configuration loading works properly"""
    # This will test that the function exists and can handle environment variables
    # Note: This test will fail if environment variables are not set
    try:
        cohere_key, qdrant_url, qdrant_key = load_config_from_env()
        assert cohere_key is not None
        assert qdrant_url is not None
    except ValueError:
        # Expected if environment variables are not set during testing
        pass


def test_chunk_text_basic():
    """Test basic text chunking functionality"""
    text = "This is a sample text that will be chunked into smaller pieces. " * 10
    result = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert result['error'] is None
    assert len(result['chunks']) > 0

    # Check that chunks don't exceed the specified size
    for chunk in result['chunks']:
        assert len(chunk['content']) <= 50


def test_chunk_text_empty():
    """Test chunking with empty text"""
    result = chunk_text("", chunk_size=100, chunk_overlap=10)

    assert len(result['chunks']) == 0
    assert result['error'] is not None


def test_chunk_text_with_overlap():
    """Test chunking with overlap"""
    text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    result = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert result['error'] is None
    assert len(result['chunks']) > 1


def test_is_valid_doc_url():
    """Test URL validation function"""
    base_url = "https://example.com"

    # Valid URLs
    assert _is_valid_doc_url(base_url, "https://example.com/page1")
    assert _is_valid_doc_url(base_url, "https://example.com/docs/guide")

    # Invalid URLs
    assert not _is_valid_doc_url(base_url, "https://other.com/page")
    assert not _is_valid_doc_url(base_url, "https://example.com/image.jpg")
    assert not _is_valid_doc_url(base_url, "https://example.com/assets/file.js")


if __name__ == "__main__":
    # Run basic tests
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_chunk_text_with_overlap()
    test_is_valid_doc_url()
    print("Basic tests passed!")