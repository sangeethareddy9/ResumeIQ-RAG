"""
src/models.py

Shared data models for the resume screener.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"


class ParseStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED_TYPE = "unsupported_type"
    CORRUPT_OR_UNREADABLE = "corrupt_or_unreadable"
    ERROR = "error"


class ParsedDocument(BaseModel):
    filename: str
    file_type: Optional[FileType] = None
    status: ParseStatus
    raw_text: str = ""
    cleaned_text: str = ""
    char_count: int = 0
    error_message: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        return self.status == ParseStatus.OK and self.char_count >= 50


class JobDescription(BaseModel):
    source: str = Field(description="'pasted_text' or original filename")
    raw_text: str
    cleaned_text: str
    extracted_skills: list[str] = Field(default_factory=list)