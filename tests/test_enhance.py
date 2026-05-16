"""
Tests for image enhancement.
"""

import pytest


@pytest.mark.asyncio
async def test_enhance_endpoint_exists(client):
    """Test that enhance endpoint exists."""
    # Just verify the endpoint is registered
    response = await client.post("/api/v1/enhance")
    # Should fail with 422 (missing file) not 404
    assert response.status_code != 404
