"""
File handling utilities for Vision_S8.

Manages file uploads, downloads, and temporary storage.
"""

import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import aiofiles

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("vision_s8.file_handler")


class FileHandler:
    """Handles file operations for uploads and outputs."""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"}

    def __init__(
        self,
        upload_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize the file handler."""
        self.upload_dir = upload_dir or settings.upload_dir
        self.output_dir = output_dir or settings.output_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, original_name: str, prefix: str = "") -> str:
        """Generate a unique filename."""
        ext = Path(original_name).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        prefix_str = f"{prefix}_" if prefix else ""
        return f"{prefix_str}{timestamp}_{unique_id}{ext}"

    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Validate an uploaded file.

        Args:
            filename: Original filename
            file_size: File size in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower()

        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File type '{ext}' not allowed. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        if file_size > settings.max_upload_size_bytes:
            max_mb = settings.max_upload_size_mb
            return False, f"File size exceeds maximum allowed ({max_mb}MB)"

        return True, ""

    async def save_upload(
        self,
        file: BinaryIO,
        filename: str,
        prefix: str = "",
    ) -> Path:
        """
        Save an uploaded file.

        Args:
            file: File-like object
            filename: Original filename
            prefix: Optional prefix for the saved filename

        Returns:
            Path to saved file
        """
        safe_filename = self._generate_filename(filename, prefix)
        save_path = self.upload_dir / safe_filename

        async with aiofiles.open(save_path, "wb") as f:
            content = file.read() if hasattr(file, "read") else file
            if isinstance(content, bytes):
                await f.write(content)
            else:
                await f.write(content)

        logger.debug(f"Saved upload: {save_path}")
        return save_path

    async def save_upload_async(
        self,
        content: bytes,
        filename: str,
        prefix: str = "",
    ) -> Path:
        """
        Save uploaded file content asynchronously.

        Args:
            content: File content as bytes
            filename: Original filename
            prefix: Optional prefix

        Returns:
            Path to saved file
        """
        safe_filename = self._generate_filename(filename, prefix)
        save_path = self.upload_dir / safe_filename

        async with aiofiles.open(save_path, "wb") as f:
            await f.write(content)

        logger.debug(f"Saved upload: {save_path}")
        return save_path

    async def save_output(
        self,
        content: bytes,
        filename: str,
        prefix: str = "",
    ) -> Path:
        """
        Save an output file.

        Args:
            content: File content
            filename: Desired filename
            prefix: Optional prefix

        Returns:
            Path to saved file
        """
        safe_filename = self._generate_filename(filename, prefix)
        save_path = self.output_dir / safe_filename

        async with aiofiles.open(save_path, "wb") as f:
            await f.write(content)

        logger.debug(f"Saved output: {save_path}")
        return save_path

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file.

        Args:
            file_path: Path to file

        Returns:
            True if deleted successfully
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old files from upload and output directories.

        Args:
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        for directory in [self.upload_dir, self.output_dir]:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                    if self.delete_file(file_path):
                        deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old files")
        return deleted_count

    def get_output_url(self, file_path: Path) -> str:
        """
        Get the relative URL for an output file.

        Args:
            file_path: Path to output file

        Returns:
            Relative URL string
        """
        return f"/outputs/{file_path.name}"


# Singleton instance
_file_handler: FileHandler | None = None


def get_file_handler() -> FileHandler:
    """Get the file handler singleton."""
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler()
    return _file_handler
