from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import TextChunk
from .config import ArasConfig


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    status: str
    provider: str


class EmbeddingArtifactBuilder:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config

    def build_pending_records(self, *, chunks: list[TextChunk], slug: str) -> tuple[list[EmbeddingRecord], Path]:
        records = [
            EmbeddingRecord(chunk_id=chunk.chunk_id, status='pending_embedding', provider='unconfigured')
            for chunk in chunks
        ]
        path = self.config.paths.processed_dir / f"{slug}-embeddings.json"
        path.write_text(
            json.dumps(
                [
                    {
                        'chunk_id': record.chunk_id,
                        'status': record.status,
                        'provider': record.provider,
                    }
                    for record in records
                ],
                ensure_ascii=False,
                indent=2,
            )
            + '\n'
        )
        return records, path
