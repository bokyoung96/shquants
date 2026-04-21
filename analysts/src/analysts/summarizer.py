from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import ArasConfig
from .domain import AnalystSummary, ExtractionPacket


class CodexRunner(Protocol):
    def run(self, *, prompt: str, schema: dict, base_dir: Path, config: ArasConfig, image_paths: list[Path]) -> dict: ...


@dataclass(frozen=True)
class SubprocessCodexRunner:
    def run(self, *, prompt: str, schema: dict, base_dir: Path, config: ArasConfig, image_paths: list[Path]) -> dict:
        with tempfile.TemporaryDirectory(dir=config.paths.state_dir) as tmp_dir:
            tmp_path = Path(tmp_dir)
            schema_path = tmp_path / "schema.json"
            output_path = tmp_path / "output.json"
            schema_path.write_text(json.dumps(schema, indent=2) + "\n")
            command = [
                config.summary.cli_command,
                "exec",
                "-C",
                str(base_dir),
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "-m",
                config.summary.model,
                "-c",
                f'model_reasoning_effort="{config.summary.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            for image_path in image_paths:
                command.extend(["-i", str(image_path)])
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
                timeout=120,
            )
            if not output_path.exists():
                raise RuntimeError(f"Codex summary output missing: {completed.stderr.strip()}")
            return json.loads(output_path.read_text())


@dataclass(frozen=True)
class CodexAnalystSummarizer:
    config: ArasConfig
    base_dir: Path
    runner: CodexRunner = SubprocessCodexRunner()

    def summarize(self, *, packet: ExtractionPacket, lane: str, topic: str) -> AnalystSummary:
        payload = self.runner.run(
            prompt=self._build_prompt(packet=packet, lane=lane, topic=topic),
            schema=self._output_schema(),
            base_dir=self.base_dir,
            config=self.config,
            image_paths=[self.base_dir / path for path in packet.page_previews],
        )
        return AnalystSummary(
            lane=payload["lane"],
            topic=payload["topic"],
            headline=payload["headline"],
            executive_summary=payload["executive_summary"],
            key_points=list(payload["key_points"]),
            key_numbers=list(payload["key_numbers"]),
            risks=list(payload["risks"]),
            confidence=payload["confidence"],
            cited_pages=list(payload["cited_pages"]),
            follow_up_questions=list(payload["follow_up_questions"]),
        )

    @staticmethod
    def lane_plan(packet: ExtractionPacket) -> list[tuple[str, str]]:
        route_topics: dict[str, str] = {}
        for hint in packet.route_hints:
            if ":" not in hint:
                continue
            lane, topic = hint.split(":", 1)
            route_topics.setdefault(lane, topic)
        return [
            ("sector", route_topics.get("sector", "general")),
            ("macro", route_topics.get("macro", "general")),
        ]

    def _build_prompt(self, *, packet: ExtractionPacket, lane: str, topic: str) -> str:
        return (
            f"You are a concise but specific {lane} expert analyst agent. Summarize this report cheaply and practically. "
            f"Use the same language as the source when obvious. If the report is not strongly relevant to your lane, say so briefly instead of forcing a view. "
            f"Prefer concrete numbers, chart/table takeaways, and page-aware evidence over generic phrasing. "
            f"Keep key_points and risks short (max {self.config.summary.max_key_points}).\n\n"
            f"Lane: {lane}\n"
            f"Topic hint: {topic}\n"
            f"Channel: {packet.report_channel}\n"
            f"Message ID: {packet.message_id}\n"
            f"Published at: {packet.published_at}\n"
            f"Extraction quality: {packet.extraction_quality}\n"
            f"Extraction reason: {packet.extraction_reason}\n"
            f"Route hints: {', '.join(packet.route_hints) or 'none'}\n"
            f"Entities: {', '.join(packet.entities) or 'none'}\n"
            f"Tickers: {', '.join(packet.tickers) or 'none'}\n"
            f"Important pages: {', '.join(str(page) for page in packet.important_pages) or 'none'}\n"
            f"Attached preview images correspond to selected important pages when available. Inspect them for charts/tables/layout cues.\n\n"
            f"Source text excerpt:\n{packet.text_excerpt}\n"
        )

    @staticmethod
    def _output_schema() -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "lane",
                "topic",
                "headline",
                "executive_summary",
                "key_points",
                "key_numbers",
                "risks",
                "confidence",
                "cited_pages",
                "follow_up_questions",
            ],
            "properties": {
                "lane": {"type": "string"},
                "topic": {"type": "string"},
                "headline": {"type": "string"},
                "executive_summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "key_numbers": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "risks": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "cited_pages": {"type": "array", "items": {"type": "integer"}, "maxItems": 6},
                "follow_up_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            },
        }
