import EChartsReact from "echarts-for-react";
import { motion } from "framer-motion";

import { formatMoney, formatPercent } from "../lib/format";
import type { DashboardPayload, ResearchFocus } from "../lib/types";

type PerformanceStripProps = {
  dashboard: DashboardPayload;
  focus: ResearchFocus;
  onFocusChange: (focus: ResearchFocus) => void;
};

const STRATEGY_COLORS = ["#f0a44b", "#e3c06b", "#7cb8d8", "#c98f7d", "#8fa77f", "#bea1d8"];

function focusLabel(focus: ResearchFocus, dashboard: DashboardPayload) {
  if (focus.kind === "strategy") {
    const context = dashboard.context[focus.runId];
    return `Focus: Strategy · ${context?.label ?? focus.runId}`;
  }

  if (focus.kind === "sector") {
    return `Focus: Sector · ${focus.sectorName}`;
  }

  return "Focus: All selected";
}

function normalizeFocusLabel(value: string) {
  return value.replace(/\s(?:쨌|夷?)\s/g, " · ");
}

function formatLaunchText(value: string | null) {
  return value ?? "n/a";
}

function formatLaunchMoney(value: number | null) {
  return value == null ? "n/a" : formatMoney(value);
}

function formatCostValue(value: number | null) {
  return value == null ? "n/a" : formatPercent(value, 2);
}

function buildCostSummary(launch: DashboardPayload["launch"]) {
  return [
    `Fee ${formatCostValue(launch.fee)}`,
    `Sell tax ${formatCostValue(launch.sellTax)}`,
    `Slippage ${formatCostValue(launch.slippage)}`,
  ].join(" / ");
}

function benchmarkSummary(dashboard: DashboardPayload) {
  const benchmark = dashboard.launch.benchmark;

  if (!benchmark) {
    return "n/a";
  }

  if (benchmark.kind === "shared" && benchmark.shared) {
    return benchmark.shared.name;
  }

  if (benchmark.strategies.length === 0) {
    return "Strategy-specific";
  }

  return benchmark.strategies.map((entry) => `${entry.label}: ${entry.benchmark.name}`).join(" | ");
}

export function PerformanceStrip({ dashboard, focus, onFocusChange }: PerformanceStripProps) {
  const selectedRuns = dashboard.selectedRunIds.map((runId, index) => {
    const run = dashboard.availableRuns.find((entry) => entry.run_id === runId);
    const metric = dashboard.metrics[runId];
    const context = dashboard.context[runId];
    const benchmark =
      dashboard.performance.benchmarks.find((entry) => entry.runId === runId) ??
      (dashboard.mode === "single" && dashboard.performance.benchmark
        ? {
            runId,
            label: context?.benchmark.name ?? "Benchmark",
            points: dashboard.performance.benchmark,
          }
        : undefined);
    const color = STRATEGY_COLORS[index % STRATEGY_COLORS.length];

    return {
      runId,
      color,
      label: context?.label ?? run?.label ?? runId,
      strategy: context?.strategy ?? run?.strategy ?? runId,
      benchmarkLabel: benchmark?.label ?? context?.benchmark.name ?? "Benchmark",
      metric,
      benchmark,
      series: dashboard.performance.series.find((entry) => entry.runId === runId),
    };
  });

  const summarySeries = selectedRuns.flatMap((run) => {
    const strategySeries = run.series
      ? [
          {
            name: run.label,
            type: "line" as const,
            data: run.series.points.map((point) => [point.date, point.value]),
            showSymbol: false,
            smooth: true,
            lineStyle: { width: 2.6, color: run.color },
            itemStyle: { color: run.color },
            emphasis: { focus: "series" as const },
          },
        ]
      : [];

    const benchmarkSeries = run.benchmark
      ? [
          {
            name: run.benchmarkLabel,
            type: "line" as const,
            data: run.benchmark.points.map((point) => [point.date, point.value]),
            showSymbol: false,
            smooth: true,
            lineStyle: { width: 1.8, type: "dashed" as const, color: run.color, opacity: 0.65 },
            itemStyle: { color: run.color, opacity: 0.65 },
            emphasis: { focus: "series" as const },
          },
        ]
      : [];

    return [...strategySeries, ...benchmarkSeries];
  });

  const chartOption = {
    backgroundColor: "transparent",
    animationDuration: 450,
    animationEasing: "cubicOut",
    grid: {
      left: 10,
      right: 24,
      top: 18,
      bottom: 28,
      containLabel: true,
    },
    tooltip: {
      trigger: "axis" as const,
      valueFormatter: (value: number) => formatMoney(value),
      backgroundColor: "rgba(15, 18, 21, 0.96)",
      borderColor: "rgba(240, 164, 75, 0.22)",
      textStyle: { color: "#f7f0e7" },
    },
    legend: {
      top: 0,
      textStyle: {
        color: "#bdaea1",
        fontFamily: "inherit",
      },
    },
    xAxis: {
      type: "time" as const,
      axisLine: { lineStyle: { color: "rgba(247, 240, 231, 0.18)" } },
      axisLabel: { color: "#bdaea1" },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: {
        color: "#bdaea1",
        formatter: (value: number) => formatMoney(value),
      },
      splitLine: {
        lineStyle: { color: "rgba(247, 240, 231, 0.08)" },
      },
    },
    series: summarySeries,
  };

  const drawdownSeries = selectedRuns
    .map((run) => {
      const series = dashboard.performance.drawdowns.find((entry) => entry.runId === run.runId);
      if (!series) {
        return null;
      }
      return {
        name: `${run.label} drawdown`,
        type: "line" as const,
        data: series.points.map((point) => [point.date, point.value]),
        showSymbol: false,
        lineStyle: { width: 1.6 },
      };
    })
    .filter((entry): entry is { name: string; type: "line"; data: [string, number][]; showSymbol: boolean; lineStyle: { width: number } } => Boolean(entry));

  const drawdownOption = {
    backgroundColor: "transparent",
    color: STRATEGY_COLORS,
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "cross" },
      formatter: (params: any) => params.map((entry: any) => `${entry.seriesName}: ${formatPercent(entry.data[1])}`).join("<br/>"),
    },
    legend: { show: false },
    grid: { left: 8, right: 18, top: 8, bottom: 12, containLabel: true },
    xAxis: {
      type: "time" as const,
      axisLine: { lineStyle: { color: "rgba(247, 240, 231, 0.18)" } },
      axisLabel: { color: "#bdaea1" },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%`, color: "#bdaea1" },
      splitLine: { lineStyle: { color: "rgba(247, 240, 231, 0.08)" } },
    },
    series: drawdownSeries,
  };

  return (
    <motion.section
      className="workspace-strip performance-strip"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="workspace-strip-copy">
        <div>
          <p className="section-label">Comparison plane</p>
          <h2>{dashboard.mode === "single" ? "Single strategy view" : "Multi strategy comparison"}</h2>
        </div>
        <p className="workspace-summary">
          Each selected strategy is paired with its own benchmark overlay. Click a strategy row to inspect it without
          leaving the page.
        </p>
      </div>

      <div className="workspace-strip-grid workspace-strip-grid--wide">
        <div className="workspace-strip-main">
          <section className="workspace-metadata-rail workspace-metadata-rail--hero" aria-label="Launch metadata">
            <div className="workspace-metadata-head">
              <span className="section-label">Launch metadata</span>
              <span>{dashboard.launch.asOfDate ? `As of ${dashboard.launch.asOfDate}` : "Snapshot context"}</span>
            </div>
            <dl className="workspace-metadata-grid">
              <div>
                <dt>Configured start</dt>
                <dd>{formatLaunchText(dashboard.launch.configuredStartDate)}</dd>
              </div>
              <div>
                <dt>Configured end</dt>
                <dd>{formatLaunchText(dashboard.launch.configuredEndDate)}</dd>
              </div>
              <div>
                <dt>Benchmark</dt>
                <dd>{benchmarkSummary(dashboard)}</dd>
              </div>
              <div>
                <dt>Costs</dt>
                <dd>{buildCostSummary(dashboard.launch)}</dd>
              </div>
            </dl>
          </section>

          <div className="workspace-chart-stack">
            <motion.div className="workspace-chart-plane workspace-chart-plane--hero" whileHover={{ scale: 1.003 }}>
              <EChartsReact option={chartOption} style={{ height: 360, width: "100%" }} />
            </motion.div>
            <motion.div className="workspace-chart-plane workspace-chart-plane--drawdown" whileHover={{ scale: 1.0 }}>
              <EChartsReact option={drawdownOption} style={{ height: 140, width: "100%" }} />
            </motion.div>
          </div>
        </div>

        <div className="workspace-metrics workspace-metrics--stack workspace-metrics--stack-side">
          <div className="focus-banner">
            <span className="section-label">Detail focus</span>
            <strong>{normalizeFocusLabel(focusLabel(focus, dashboard))}</strong>
          </div>

          {selectedRuns.map((run) => {
            const isFocused = focus.kind === "strategy" && focus.runId === run.runId;
            return (
              <button
                key={run.runId}
                type="button"
                className={`workspace-metric-line workspace-metric-line--button ${isFocused ? "is-focused" : ""}`}
                onClick={() => onFocusChange({ kind: "strategy", runId: run.runId })}
                aria-pressed={isFocused}
                aria-label={`Focus strategy ${run.label}`}
              >
                <div className="workspace-metric-head">
                  <strong>{run.label}</strong>
                  <span>{run.strategy}</span>
                </div>
                <div className="workspace-metric-row">
                  <span>CAGR {run.metric ? formatPercent(run.metric.cagr) : "n/a"}</span>
                  <span>Sharpe {run.metric ? run.metric.sharpe.toFixed(2) : "n/a"}</span>
                </div>
                <div className="workspace-metric-row">
                  <span>Max drawdown {run.metric ? formatPercent(run.metric.maxDrawdown) : "n/a"}</span>
                  <span>{run.benchmarkLabel}</span>
                </div>
                <div className="workspace-metric-row workspace-metric-accent">
                  <span>Final equity</span>
                  <span>{run.metric ? formatMoney(run.metric.finalEquity) : "n/a"}</span>
                </div>
                <span className="workspace-metric-action">Focus strategy {run.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </motion.section>
  );
}
