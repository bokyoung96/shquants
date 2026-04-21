from analysts.chunking import TextChunker
from analysts.config import build_config



def test_chunker_splits_long_text_deterministically(tmp_path):
    config = build_config(tmp_path)
    chunker = TextChunker(config)
    text = 'A' * 2500

    chunks, path = chunker.chunk_text(text=text, slug='report-1')

    assert len(chunks) >= 2
    assert chunks[0].chunk_id == 'report-1-chunk-0'
    assert path.exists()
