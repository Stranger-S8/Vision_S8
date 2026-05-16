"""
Pytest configuration for Vision_S8 tests.
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

# Set test environment
os.environ["GEMINI_API_KEY"] = "test_api_key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def app():
    """Create a test application instance."""
    from vision_s8.main import app
    from vision_s8.models.database import init_database, close_database

    # Initialize test database
    await init_database("sqlite+aiosqlite:///:memory:")

    yield app

    # Cleanup
    await close_database()


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def test_image() -> Image.Image:
    """Create a simple test image."""
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    # Add some variation
    for x in range(100, 200):
        for y in range(100, 200):
            img.putpixel((x, y), (200, 100, 100))
    return img


@pytest.fixture
def test_image_bytes(test_image: Image.Image) -> bytes:
    """Get test image as bytes."""
    import io
    buffer = io.BytesIO()
    test_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def test_image_path(test_image: Image.Image, tmp_path: Path) -> Path:
    """Save test image to a temporary file."""
    path = tmp_path / "test_image.jpg"
    test_image.save(path, format="JPEG")
    return path


@pytest.fixture
def multiple_test_images() -> list[Image.Image]:
    """Create multiple test images for A/B testing."""
    images = []
    colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255)]
    for color in colors:
        img = Image.new("RGB", (1000, 1000), color=color)
        images.append(img)
    return images
