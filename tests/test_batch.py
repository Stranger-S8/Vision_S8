"""
Tests for batch processing.
"""

import io
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_list_batch_jobs(client):
    """Test listing batch jobs."""
    response = await client.get("/api/v1/batch")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "jobs" in data["data"]


@pytest.mark.asyncio
async def test_batch_upload_no_files(client):
    """Test batch upload with no files."""
    response = await client.post("/api/v1/batch/upload")

    # Should fail validation
    assert response.status_code in [400, 422]
