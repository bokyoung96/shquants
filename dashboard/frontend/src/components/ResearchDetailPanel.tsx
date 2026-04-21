import { motion } from "framer-motion";

import { formatPercent } from "../lib/format";
import type { DashboardPayload, ResearchFocus, SeriesPoint } from "../lib/types";

type ResearchDetailPanelProps = {
  dashboard: DashboardPayload;
  focus: ResearchFocus;
};

function visibleRunIds(dashboard: DashboardPayload, focus: ResearchFocus) {
  if (focus.kind === "strategy" && dashboard.selectedRunIds.includes(focus.runId)) {
    return [focus.runId];
  }

  return dashboard.selectedRunIds;
}

function focusSummary(focus: ResearchFocus, dashboard: DashboardPayload) {
  if (focus.kind === "strategy") {
    return dashboard.context[focus.runId]?.label ?? focus.runId;
  }

  if (focus.kind === "sector") {
    return `Sector · ${focus.sectorName}`;
  }

  return "All selected";
}

function flattenEpisodes(dashboard: DashboardPayload, runIds: string[]) {
  return runIds.flatMap((runId) => {
    const label = dashboard.context[runId]?.label ?? runId;
    return (dashboard.research.drawdownEpisodes[runId] ?? []).map((episode) => ({
      runId,
      label,
      episode,
    }));
  });
}

function formatNumberValue(value: number, digits = 2) {
  return value.toFixed(digits);
}

function formatMetricPercent(value: number | undefined, fractionDigits = 1) {
  return value == null ? "n/a" : formatPercent(value, fractionDigits);
}

function formatMetricNumber(value: number | undefined, digits = 2) {
  return value == null ? "n/a" : formatNumberValue(value, digits);
}

function formatRewardRisk(value: number) {
  if (!Number.isFinite(value)) {
    return value === 0 ? "n/a" : "∞";
  }
  return `${value.toFixed(2)}:1`;
}

function computeValueDiffs(series?: SeriesPoint[]) {
  if (!series || series.length < 2) {
    return [];
  }
  const values: number[] = [];
  for (let index = 1; index < series.length; index += 1) {
    const previous = series[index - 1].value;
    const current = series[index].value;
    if (Number.isFinite(previous) && Number.isFinite(current)) {
      values.push(current - previous);
    }
  }
  return values;
}

export function ResearchDetailPanel({ dashboard, focus }: ResearchDetailPanelProps) {
  const runIds = visibleRunIds(dashboard, focus);
  const episodes = flattenEpisodes(dashboard, runIds);
  const metricRunId =
    focus.kind === "strategy" && dashboard.metrics[focus.runId] ? focus.runId : runIds[0] ?? "";
  const metric = metricRunId ? dashboard.metrics[metricRunId] : undefined;
  const hasMetric = Boolean(metric);
  const seriesPoints = dashboard.performance.series.find((series) => series.runId === metricRunId)?.points;
  const diffs = computeValueDiffs(seriesPoints);
  const positiveDiffs = diffs.filter((value) => value > 0);
  const negativeDiffs = diffs.filter((value) => value < 0);
  const hitRate = diffs.length ? positiveDiffs.length / diffs.length : 0;
  const avgGain = positiveDiffs.length > 0 ? positiveDiffs.reduce((sum, value) => sum + value, 0) / positiveDiffs.length : 0;
  const avgLoss =
    negativeDiffs.length > 0
      ? negativeDiffs.reduce((sum, value) => sum + Math.abs(value), 0) / negativeDiffs.length
      : 0;
  const profitRisk = avgLoss === 0 ? (avgGain > 0 ? Infinity : 0) : avgGain / avgLoss;
  const openEpisodes = episodes.filter((entry) => !entry.episode.recovered);
  const longestOpen = [...openEpisodes].sort(
    (left, right) => right.episode.durationDays - left.episode.durationDays,
  )[0];
  const fallbackEpisode = [...episodes].sort((left, right) => right.episode.drawdown - left.episode.drawdown)[0];
  const toughestEpisode = longestOpen ?? fallbackEpisode;
  const metricOrder: Array<[string, string]> = [
    ["Cumulative return", formatMetricPercent(metric?.cumulativeReturn, 1)],
    ["Max drawdown", formatMetricPercent(metric?.maxDrawdown, 1)],
    ["Sharpe", formatMetricNumber(metric?.sharpe, 2)],
    ["Calmar", formatMetricNumber(metric?.calmar, 2)],
    ["Information Ratio", formatMetricNumber(metric?.informationRatio, 2)],
    ["Hit Rate", hasMetric && diffs.length ? formatPercent(hitRate, 1) : "n/a"],
    ["Profit / Risk", hasMetric ? formatRewardRisk(profitRisk) : "n/a"],
  ];

  return (
    <motion.section
      className="detail-section research-detail-panel"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.42, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="detail-section-copy">
        <p className="section-label">Details</p>
        <h2>Research details</h2>
        <p className="workspace-summary">Key metrics and drawdown context for the current focus.</p>
      </div>

      <div className="focus-banner focus-banner--inline">
        <span className="section-label">Current focus</span>
        <strong>{focusSummary(focus, dashboard)}</strong>
      </div>

      <div className="detail-panel-grid">
        <div className="detail-metric-strip detail-metric-strip--row">
          {metricOrder.map(([label, value]) => (
            <div key={label} className="detail-metric-chip">
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      </div>

      <div className="detail-note">
        <span className="section-label">Toughest drawdown</span>
        <strong>
          {toughestEpisode
            ? `${toughestEpisode.label}: ${(toughestEpisode.episode.drawdown * 100).toFixed(1)}% · ${toughestEpisode.episode.durationDays}d`
            : "No drawdowns recorded"}
        </strong>
        <p>
          {toughestEpisode
            ? toughestEpisode.episode.recovered
              ? `Recovered in ${toughestEpisode.episode.recoveryDays ?? "?"} days (trough ${toughestEpisode.episode.trough}).`
              : `Still open since ${toughestEpisode.episode.start}.`
            : "Awaiting drawdown data."}
        </p>
      </div>

    </motion.section>
  );
}
