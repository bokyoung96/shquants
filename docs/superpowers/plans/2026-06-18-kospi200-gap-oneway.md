# KOSPI200 Gap One-Way Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the `etc/momentum` plot, Excel workbook, and report for KOSPI200 gap-up days over three overlapping windows: 2025-01 through current data, 2026-01 through current data, and sessions from 2026-05-27 onward.

**Architecture:** Use one standalone reproducible script that reads the local KOSPI200 1-minute Excel source, derives session-level gap and intraday-flow metrics, writes all analysis tables to Excel, draws one multi-subplot PNG, and writes a Korean Markdown report. The script keeps data definitions explicit so the artifacts can be regenerated.

**Tech Stack:** Python 3.11, pandas, matplotlib, openpyxl.

---

### Task 1: Build Reproducible Analysis Script

**Files:**
- Create: `etc/momentum/build_kospi200_gap_oneway_report.py`

- [ ] Load `etc/data/sidecar/KOSPI200 INDEX(1분).xlsx` by globbing for `*KOSPI200*INDEX*.xlsx`.
- [ ] Normalize datetime, OHLC, and volume columns.
- [ ] Build daily sessions with previous-session close, first available minute open, full-day close, gap percentage, intraday return, MFE, MAE, range, close location, and full-session flag.
- [ ] Define overlapping target periods: `2025-01~current`, `2026-01~current`, and `2026-05-27~current`.
- [ ] Define threshold events for `gap_open_pct >= +1%`, `>= +2%`, `>= +3%`, `>= +4%`, and `>= +5%`.
- [ ] Define one-way-up days as full sessions with positive open-to-close return, close in the top 30% of the day range, and no worse than -0.4% open-to-low drawdown.
- [ ] Compute exit-time returns from session open to 09:30, 10:00, 10:30, 11:00, 11:30, 12:00, 13:00, 13:30, 14:00, 14:30, 15:00, 15:20, and 15:30.

### Task 2: Generate Deliverables

**Files:**
- Modify: `etc/momentum/momentum_strategy_graph.png`
- Modify: `etc/momentum/momentum_strategy_results.xlsx`
- Modify: `etc/momentum/momentum_strategy_report.md`

- [ ] Write Excel sheets for README, data quality, daily sessions, gap days, bucket summary, threshold summary, one-way summary, exit summary, best exits, event-level best exits, and checkpoint returns.
- [ ] Draw one PNG containing multiple subplots: event counts, one-way ratio, average intraday flow, exit-time mean returns, period flow, and gap versus close-return scatter.
- [ ] Write a Korean Markdown report with definitions, sample warnings, one-way composition, flow interpretation, best exit-time tables, and artifact list.

### Task 3: Verify

**Files:**
- Inspect: `etc/momentum/momentum_strategy_report.md`
- Inspect: `etc/momentum/momentum_strategy_results.xlsx`
- Inspect: `etc/momentum/momentum_strategy_graph.png`

- [ ] Run `python etc/momentum/build_kospi200_gap_oneway_report.py`.
- [ ] Confirm the workbook has the required sheets and covers all three requested periods.
- [ ] Confirm the report references KOSPI200, +1% to +5% thresholds, one-way composition, flow, and ideal exit times.
- [ ] Confirm the PNG exists and has nonzero dimensions.
