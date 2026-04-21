from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig


@dataclass(frozen=True)
class GraphifyUpdateResult:
    corpus_dir: Path
    manifest_path: Path
    report_count: int
    graphify_invoked: bool


class GraphifyCorpusBuilder:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config

    def update(self) -> GraphifyUpdateResult:
        processed_dir = self.config.paths.processed_dir
        corpus_dir = processed_dir / 'graphify-corpus'
        corpus_dir.mkdir(parents=True, exist_ok=True)
        summaries = sorted(processed_dir.glob('report-*-summary.json'))
        manifest_items = []
        for summary_path in summaries:
            payload = json.loads(summary_path.read_text())
            report_id = summary_path.stem.replace('-summary', '')
            doc_path = corpus_dir / f'{report_id}.md'
            lines = [
                f"# {payload['report_title']}",
                '',
                f"- message_id: {payload['message_id']}",
                f"- raw_pdf_path: {payload['raw_pdf_path']}",
                f"- important_pages: {', '.join(map(str, payload.get('important_pages', []))) or 'none'}",
                '',
            ]
            for summary in payload.get('summaries', []):
                lines += [
                    f"## {summary['lane']} expert",
                    f"- headline: {summary['headline']}",
                    f"- confidence: {summary['confidence']}",
                    f"- cited_pages: {', '.join(map(str, summary.get('cited_pages', []))) or 'none'}",
                    '',
                    summary['executive_summary'],
                    '',
                    '### key_points',
                    *[f"- {item}" for item in summary.get('key_points', [])],
                    '',
                    '### key_numbers',
                    *[f"- {item}" for item in summary.get('key_numbers', [])],
                    '',
                    '### risks',
                    *[f"- {item}" for item in summary.get('risks', [])],
                    '',
                ]
            doc_path.write_text('\n'.join(lines).rstrip() + '\n')
            manifest_items.append({
                'report_id': report_id,
                'message_id': payload['message_id'],
                'source_summary_json': str(summary_path.relative_to(self.config.paths.base_dir)),
                'graphify_doc': str(doc_path.relative_to(self.config.paths.base_dir)),
            })
        manifest_path = corpus_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest_items, ensure_ascii=False, indent=2) + '\n')

        graphify_invoked = False
        if manifest_items:
            try:
                completed = subprocess.run(
                    [sys.executable, '-m', 'graphify', str(corpus_dir), '--update', '--no-viz'],
                    cwd=self.config.paths.base_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                graphify_invoked = completed.returncode == 0
            except Exception:
                graphify_invoked = False

        return GraphifyUpdateResult(
            corpus_dir=corpus_dir,
            manifest_path=manifest_path,
            report_count=len(manifest_items),
            graphify_invoked=graphify_invoked,
        )
