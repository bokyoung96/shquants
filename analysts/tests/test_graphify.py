import json
from pathlib import Path

from analysts.config import build_config
from analysts.graphify import GraphifyCorpusBuilder



def test_graphify_corpus_builder_writes_manifest_and_docs(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    processed = config.paths.processed_dir
    processed.mkdir(parents=True, exist_ok=True)
    (processed / 'report-1-summary.json').write_text(json.dumps({
        'report_title': 'Report Title',
        'message_id': 123,
        'raw_pdf_path': 'data/raw/sample.pdf',
        'important_pages': [1, 2],
        'summaries': [
            {
                'lane': 'sector',
                'headline': 'Headline',
                'confidence': 'high',
                'cited_pages': [1, 2],
                'executive_summary': 'Summary body',
                'key_points': ['A'],
                'key_numbers': ['10%'],
                'risks': ['Risk'],
            }
        ],
    }, ensure_ascii=False, indent=2))

    result = GraphifyCorpusBuilder(config).update()

    assert result.report_count == 1
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest[0]['message_id'] == 123
    assert (result.corpus_dir / 'report-1.md').exists()
