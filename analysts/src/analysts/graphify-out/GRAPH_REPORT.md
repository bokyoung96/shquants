# Graph Report - analysts/src/analysts  (2026-04-22)

## Corpus Check
- Corpus is ~11,571 words - fits in a single context window. You may not need a graph.

## Summary
- 340 nodes · 906 edges · 10 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 294 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Gmail Pipeline|Gmail Pipeline]]
- [[_COMMUNITY_Telegram Fetching|Telegram Fetching]]
- [[_COMMUNITY_PDF Ingestion|PDF Ingestion]]
- [[_COMMUNITY_Core Summaries|Core Summaries]]
- [[_COMMUNITY_Telethon Client|Telethon Client]]
- [[_COMMUNITY_CLI Wiring|CLI Wiring]]
- [[_COMMUNITY_Watch Runner|Watch Runner]]
- [[_COMMUNITY_Gmail OAuth|Gmail OAuth]]
- [[_COMMUNITY_Runtime Config|Runtime Config]]
- [[_COMMUNITY_Raw Report Catalog|Raw Report Catalog]]

## God Nodes (most connected - your core abstractions)
1. `ArasConfig` - 36 edges
2. `ArasPipeline` - 29 edges
3. `ReportRecord` - 26 edges
4. `SqliteArasStore` - 25 edges
5. `main()` - 20 edges
6. `TelegramFetcher` - 20 edges
7. `GmailSourcePipeline` - 19 edges
8. `GmailStore` - 19 edges
9. `Gmail source ingestion package.` - 18 edges
10. `PdfIngestionPipeline` - 16 edges

## Surprising Connections (you probably didn't know these)
- `build_default_pipeline()` --calls--> `from_fixture_path()`  [INFERRED]
  analysts/src/analysts/cli.py → analysts/src/analysts/sources/telegram/client.py
- `build_default_pipeline()` --calls--> `SqliteArasStore`  [INFERRED]
  analysts/src/analysts/cli.py → analysts/src/analysts/storage.py
- `build_default_pipeline()` --calls--> `ArasPipeline`  [INFERRED]
  analysts/src/analysts/cli.py → analysts/src/analysts/pipeline.py
- `build_default_pipeline()` --calls--> `TelethonChannelClient`  [INFERRED]
  analysts/src/analysts/cli.py → analysts/src/analysts/sources/telegram/fetcher.py
- `build_watch_runner()` --calls--> `SqliteArasStore`  [INFERRED]
  analysts/src/analysts/cli.py → analysts/src/analysts/storage.py

## Communities

### Community 0 - "Gmail Pipeline"
Cohesion: 0.07
Nodes (37): _clean(), ensure(), find(), name(), CanonicalDocument, GmailAttachmentRecord, GmailCandidateDocument, GmailMessageRecord (+29 more)

### Community 1 - "Telegram Fetching"
Cohesion: 0.1
Nodes (19): ReportRecord, _downloadable_title(), DownloadableMessage, _extract_downloadable_message(), _extract_pdf_message(), _extract_supported_pdf_payload(), FetchBatch, _format_timestamp() (+11 more)

### Community 2 - "PDF Ingestion"
Cohesion: 0.12
Nodes (21): TextChunk, TextChunker, ArasConfig, ExtractionPacket, ParsedDocument, EmbeddingArtifactBuilder, EmbeddingRecord, _clean_text() (+13 more)

### Community 3 - "Core Summaries"
Cohesion: 0.11
Nodes (22): AnalystSummary, ParseQuality, PipelineExecution, PipelineRunSummary, RouteDecision, DocumentParser, _extract_entities(), _extract_text() (+14 more)

### Community 4 - "Telethon Client"
Cohesion: 0.1
Nodes (14): _adapt_message(), auth_login(), _extract_document_payload(), FixtureTelegramClient, from_fixture_path(), _isolated_session_path(), _isolated_sync_client(), _now_like() (+6 more)

### Community 5 - "CLI Wiring"
Cohesion: 0.15
Nodes (22): build_arg_parser(), build_default_pipeline(), build_gmail_source_pipeline(), build_watch_runner(), configure_watch_logger(), main(), _merge_watch_results(), normalize_channels() (+14 more)

### Community 6 - "Watch Runner"
Cohesion: 0.14
Nodes (11): Protocol, _accepted_at(), AsyncWatchClient, AsyncWatchResult, _channel_from_message(), _channels_log_field(), MessageIngestor, _new_counts() (+3 more)

### Community 7 - "Gmail OAuth"
Cohesion: 0.22
Nodes (4): GmailApiClient, _oauth_callback_server(), _open_browser(), _token_is_valid()

### Community 8 - "Runtime Config"
Cohesion: 0.22
Nodes (8): ArasPaths, BodyCandidateRules, GmailConfig, _load_local_runtime_config(), LocalRuntimeConfig, require_telethon_config(), SummaryConfig, TelethonConfig

### Community 9 - "Raw Report Catalog"
Cohesion: 0.48
Nodes (1): RawReportCatalog

## Knowledge Gaps
- **Thin community `Raw Report Catalog`** (7 nodes): `.summarize_latest()`, `RawReportCatalog`, `.__init__()`, `.latest_report()`, `.list_reports()`, `.recent_reports()`, `._to_report()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ArasConfig` connect `PDF Ingestion` to `Gmail Pipeline`, `Telegram Fetching`, `Core Summaries`, `Telethon Client`, `CLI Wiring`, `Runtime Config`?**
  _High betweenness centrality (0.233) - this node is a cross-community bridge._
- **Why does `GmailSourcePipeline` connect `Gmail Pipeline` to `Runtime Config`, `PDF Ingestion`, `Core Summaries`, `CLI Wiring`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `ArasPipeline` connect `Core Summaries` to `Gmail Pipeline`, `Telegram Fetching`, `PDF Ingestion`, `CLI Wiring`, `Raw Report Catalog`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `ArasConfig` (e.g. with `TextChunk` and `TextChunker`) actually correct?**
  _`ArasConfig` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `ArasPipeline` (e.g. with `ArasConfig` and `AnalystSummary`) actually correct?**
  _`ArasPipeline` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `ReportRecord` (e.g. with `ExtractionArtifacts` and `SummaryReadyExtractor`) actually correct?**
  _`ReportRecord` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `SqliteArasStore` (e.g. with `ArasPipeline` and `ReportRecord`) actually correct?**
  _`SqliteArasStore` has 11 INFERRED edges - model-reasoned connections that need verification._