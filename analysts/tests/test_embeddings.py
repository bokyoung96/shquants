from analysts.chunking import TextChunk
from analysts.config import build_config
from analysts.embeddings import EmbeddingArtifactBuilder



def test_embedding_builder_writes_pending_embedding_records(tmp_path):
    config = build_config(tmp_path)
    builder = EmbeddingArtifactBuilder(config)
    chunks = [TextChunk(chunk_id='c1', chunk_index=0, text='abc', char_count=3)]

    records, path = builder.build_pending_records(chunks=chunks, slug='report-1')

    assert len(records) == 1
    assert records[0].status == 'pending_embedding'
    assert path.exists()
