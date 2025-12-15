from pydantic import BaseModel, Field
from typing import List, Optional

class MetadataRecord(BaseModel):
    title: str
    doc_type: str = "unknown"
    abstract: str

    topics: List[str] = Field(default_factory=list)
    programs: List[str] = Field(default_factory=list)
    geographies: List[str] = Field(default_factory=list)
    sensitivity: Optional[str] = None

    keywords: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)

METADATA_JSON_SCHEMA = {
  "type": "object",
  "required": ["title", "doc_type", "abstract", "topics", "programs", "geographies", "sensitivity", "keywords", "evidence_snippets"],
  "properties": {
    "title": {"type":"string"},
    "doc_type": {"type":"string"},
    "abstract": {"type":"string"},
    "topics": {"type":"array","items":{"type":"string"}},
    "programs": {"type":"array","items":{"type":"string"}},
    "geographies": {"type":"array","items":{"type":"string"}},
    "sensitivity": {"type":["string","null"]},
    "keywords": {"type":"array","items":{"type":"string"}},
    "evidence_snippets": {"type":"array","items":{"type":"string"}}
  }
}