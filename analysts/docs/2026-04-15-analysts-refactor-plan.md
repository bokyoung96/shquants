# Analysts Refactor Plan — 2026-04-15

## Scope
Refactor `analysts/` only. Preserve the active production path used by:
- `src/analysts/cli.py`
- `src/analysts/pipeline.py`
- `src/analysts/pdf_ingest.py`

## Behavior lock
- Baseline regression run before edits: `.venv/bin/pytest analysts/tests -q`
- Existing pipeline/CLI/graphify tests already cover the active path; keep them green throughout.

## Cleanup plan
1. **Dead code deletion**
   - Remove legacy modules not reachable from the current CLI/pipeline import path: `agents.py`, `summary_agents.py`, `wiki.py`, `signal.py`.
   - Remove their now-orphaned tests.
   - Trim package exports that only pointed at the deleted legacy path.

2. **Filename simplification**
   - Rename `graphify_integration.py` → `graphify.py`.
   - Rename `processed_outputs.py` → `summary_outputs.py`.
   - Update imports/tests/docs references.

3. **Pipeline clarity pass**
   - Keep `ArasPipeline` as the top-level orchestrator.
   - Make the summarize flow read like explicit stages (parse → route → ingest → summarize → write outputs) without changing behavior.
   - Keep changes small and reversible; prefer helper extraction over new abstractions.

4. **Verification**
   - Run targeted tests after cleanup edits.
   - Run the full analysts test suite before claiming completion.

## Explicit non-goals
- No new dependencies.
- No behavior changes to the active fetch/summarize/graphify path.
- Do not remove code that is still used by the active analyst pipeline.
