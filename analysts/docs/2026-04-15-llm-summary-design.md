# Token-Cheap LLM Summary Design

## Goal
Upgrade `analysts/` so downloaded Telegram PDFs are summarized by an actual analyst agent while keeping token usage low and preserving raw artifacts as the primary truth source.

## Why the current outputs failed
The previous deterministic downstream flow produced low-value summaries because live PDFs frequently decode poorly with the current extractor. When extraction fails, routing and analyst comments degrade into generic fallback output.

## Design principles
- Raw PDF remains the first-class artifact.
- Extraction must produce a compact, summary-ready text packet before any LLM call.
- Use one cheap LLM analyst call per relevant lane, not a large multi-agent conversation for every report.
- Persist inspectable artifacts under `analysts/data/processed/` so the user can review raw text, agent inputs, and agent outputs.
- Skip expensive coordination when only one lane is relevant.

## Recommended architecture
1. **Extraction lane**
   - Create a summary-ready extractor that produces:
     - `extracted.txt`
     - `summary_input.json`
   - Prefer Telegram message text/caption as a fallback signal when PDF text extraction is weak.
   - Cap text length before agent invocation.

2. **Agent summarization lane**
   - Add a `CodexAnalystSummarizer` that shells out to `codex exec` with a strict JSON schema.
   - Default model should be a cheaper frontier/mini-capable model (for example `gpt-5.4-mini`) and low/medium reasoning.
   - Run only the lanes that matter:
     - sector analyst when sector route confidence is non-empty
     - macro analyst when macro route confidence is non-empty
     - general fallback analyst when extraction is weak or no route matches

3. **Output lane**
   - Materialize concise outputs:
     - `processed/<slug>-raw-text.txt`
     - `processed/<slug>-summary-input.json`
     - `processed/<slug>-summary.json`
     - `processed/<slug>-summary.md`
   - Do not regenerate wiki/signal artifacts unless the summary is high enough quality to justify them.

## Token controls
- Hard cap extracted body characters sent to the LLM.
- Do not send the binary PDF.
- Reuse metadata and locally extracted keywords rather than asking the model to rediscover obvious facts.
- One analyst call per route max.
- No coordinator unless both sector and macro outputs exist and are non-empty.

## Analyst output contract
Each analyst response should contain:
- `lane`
- `topic`
- `headline`
- `executive_summary`
- `key_points`
- `risks`
- `confidence`
- `follow_up_questions`

## Delivery split
- Lane 1: extraction and summary-input compression
- Lane 2: codex-based analyst summarizer and route-aware lane selection
- Lane 3: processed artifact contract, CLI integration, verification, and docs
