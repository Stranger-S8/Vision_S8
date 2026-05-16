"""
Tests for image comparison.
"""

import pytest


@pytest.mark.asyncio
async def test_compare_no_competitors(client, test_image_bytes):
    """Test comparison without competitor images."""
    import io

    response = await client.post(
        "/api/v1/compare",
        files={"main_image": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
        data={"mode": "manual"},
    )

    assert response.status_code == 200
    data = response.json()
    # Should fail because no competitors provided
    assert data["success"] is False
    assert "competitor" in data["message"].lower()
