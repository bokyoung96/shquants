from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str
    char_count: int


class TextChunker:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config
        self.chunk_size = 1200
        self.chunk_overlap = 200

    def chunk_text(self, *, text: str, slug: str) -> tuple[list[TextChunk], Path]:
        chunks: list[TextChunk] = []
        start = 0
        index = 0
        normalized = text.strip()
        while normalized and start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            payload = normalized[start:end].strip()
            if payload:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{slug}-chunk-{index}",
                        chunk_index=index,
                        text=payload,
                        char_count=len(payload),
                    )
                )
            if end >= len(normalized):
                break
            start = max(0, end - self.chunk_overlap)
            index += 1
        path = self.config.paths.processed_dir / f"{slug}-chunks.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "char_count": chunk.char_count,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        return chunks, path
