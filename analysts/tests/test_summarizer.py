from pathlib import Path
import json

from analysts.config import build_config
from analysts.domain import ExtractionPacket
from analysts.summarizer import CodexAnalystSummarizer, SubprocessCodexRunner


class FakeRunner:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, *, prompt: str, schema: dict, base_dir: Path, config, image_paths) -> dict:
        self.prompts.append(prompt)
        self.image_paths = image_paths
        return {
            'lane': 'macro',
            'topic': 'general',
            'headline': '사모신용은 시스템 리스크가 아님',
            'executive_summary': '핵심 우려는 제한적이라는 요지다.',
            'key_points': ['위험 분산 구조', '은행 건전성 양호'],
            'key_numbers': ['NBFI 대출 56% 증가'],
            'risks': ['추가 부실 전이 가능성 점검 필요'],
            'confidence': 'medium',
            'cited_pages': [4, 13],
            'follow_up_questions': ['연준 스트레스 테스트 세부 수치 확인 필요'],
        }



def test_lane_plan_returns_sector_and_macro_comments() -> None:
    packet = ExtractionPacket(
        source_document_id=1,
        report_title='title',
        report_channel='DOC_POOL',
        message_id=77,
        published_at='2026-04-15T00:00:00Z',
        raw_pdf_path=Path('data/raw/sample.pdf'),
        extraction_quality='fallback',
        extraction_reason='used_title',
        preferred_text='text',
        text_excerpt='text',
        route_hints=['macro:general'],
        entities=[],
        tickers=[],
    )

    assert CodexAnalystSummarizer.lane_plan(packet) == [('sector', 'general'), ('macro', 'general')]



def test_summarizer_uses_runner_and_returns_structured_summary(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    runner = FakeRunner()
    summarizer = CodexAnalystSummarizer(config=config, base_dir=tmp_path, runner=runner)
    packet = ExtractionPacket(
        source_document_id=1,
        report_title='사모신용 이슈가 시스템 리스크가 아닌 이유',
        report_channel='DOC_POOL',
        message_id=163007,
        published_at='2026-04-15T00:34:10Z',
        raw_pdf_path=tmp_path / 'sample.pdf',
        extraction_quality='fallback',
        extraction_reason='title_fallback',
        preferred_text='긴 텍스트',
        text_excerpt='짧은 텍스트',
        route_hints=['macro:general'],
        entities=['한화투자증권'],
        tickers=[],
        page_previews=['data/processed/report-1-pages/page-1.png'],
        important_pages=[1],
    )

    summary = summarizer.summarize(packet=packet, lane='macro', topic='general')

    assert summary.lane == 'macro'
    assert summary.topic == 'general'
    assert '짧은 텍스트' in runner.prompts[0]
    assert 'Lane: macro' in runner.prompts[0]
    assert 'Important pages: 1' in runner.prompts[0]
    assert runner.image_paths[0].name == 'page-1.png'
    assert summary.key_numbers == ['NBFI 대출 56% 증가']
    assert summary.cited_pages == [4, 13]


def test_subprocess_runner_includes_skip_git_repo_check_for_isolated_dirs(tmp_path: Path, monkeypatch) -> None:
    config = build_config(tmp_path)
    runner = SubprocessCodexRunner()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "lane": "macro",
                    "topic": "general",
                    "headline": "headline",
                    "executive_summary": "summary",
                    "key_points": ["point"],
                    "key_numbers": ["42"],
                    "risks": ["risk"],
                    "confidence": "medium",
                    "cited_pages": [1],
                    "follow_up_questions": ["question"],
                }
            )
        )

        class Result:
            stderr = ""

        return Result()

    monkeypatch.setattr("analysts.summarizer.subprocess.run", fake_run)

    runner.run(
        prompt="prompt",
        schema={"type": "object"},
        base_dir=tmp_path,
        config=config,
        image_paths=[],
    )

    assert "--skip-git-repo-check" in captured["command"]
