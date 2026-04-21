import EChartsReact from "echarts-for-react";
import { motion } from "framer-motion";

import { formatPercent } from "../lib/format";
import type { DashboardPayload, ResearchFocus } from "../lib/types";

type DiagnosticStripProps = {
  dashboard: DashboardPayload;
  focus: ResearchFocus;
  onFocusChange: (focus: ResearchFocus) => void;
};

export function DiagnosticStrip({ dashboard, focus, onFocusChange }: DiagnosticStripProps) {
  const selectedRuns = dashboard.selectedRunIds.map((runId) => {
    const run = dashboard.availableRuns.find((entry) => entry.run_id === runId);
    const metric = dashboard.metrics[runId];
    const context = dashboard.context[runId];

    return {
      runId,
      label: context?.label ?? run?.label ?? runId,
      strategy: context?.strategy ?? run?.strategy ?? runId,
      cagr: metric?.cagr ?? null,
      maxDrawdown: metric?.maxDrawdown ?? null,
      informationRatio: metric?.informationRatio ?? null,
      benchmark: context?.benchmark.name ?? "Benchmark",
    };
  });

  const chartOption = {
    backgroundColor: "transparent",
    animationDuration: 380,
    grid: {
      left: 8,
      right: 18,
      top: 12,
      bottom: 24,
      containLabel: true,
    },
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "shadow" as const },
      backgroundColor: "rgba(15, 18, 21, 0.96)",
      borderColor: "rgba(240, 164, 75, 0.22)",
      textStyle: { color: "#f7f0e7" },
      valueFormatter: (value: number) => formatPercent(value),
    },
    legend: {
      top: 0,
      textStyle: { color: "#bdaea1", fontFamily: "inherit" },
    },
    xAxis: {
      type: "category" as const,
      data: selectedRuns.map((run) => run.label),
      axisLine: { lineStyle: { color: "rgba(247, 240, 231, 0.18)" } },
      axisLabel: { color: "#bdaea1" },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: {
        color: "#bdaea1",
        formatter: (value: number) => formatPercent(value),
      },
      splitLine: { lineStyle: { color: "rgba(247, 240, 231, 0.08)" } },
    },
    series: [
      {
        name: "CAGR",
        type: "bar" as const,
        data: selectedRuns.map((run) => run.cagr),
        itemStyle: { color: "#f0a44b", borderRadius: [6, 6, 0, 0] },
      },
      {
        name: "Max drawdown",
        type: "bar" as const,
        data: selectedRuns.map((run) => run.maxDrawdown),
        itemStyle: { color: "#876052", borderRadius: [6, 6, 0, 0] },
      },
    ],
  };

  return (
    <motion.section
      className="workspace-strip diagnostic-strip"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="workspace-strip-copy">
        <div>
          <p className="section-label">Selection shell</p>
          <h2>Research summary ledger</h2>
        </div>
        <p className="workspace-summary">
          Return, drawdown, and benchmark-relative efficiency stay visible above the full research workspace.
        </p>
      </div>

      <div className="workspace-strip-grid">
        <motion.div className="workspace-chart-plane workspace-chart-plane--summary" whileHover={{ scale: 1.003 }}>
          <EChartsReact option={chartOption} style={{ height: 240, width: "100%" }} />
        </motion.div>

        <div className="workspace-quiet-metrics">
          {selectedRuns.map((run) => {
            const isFocused = focus.kind === "strategy" && focus.runId === run.runId;
            return (
              <button
                key={run.runId}
                type="button"
                className={`workspace-quiet-line workspace-quiet-line--button ${isFocused ? "is-focused" : ""}`}
                onClick={() => onFocusChange({ kind: "strategy", runId: run.runId })}
                aria-label={`Inspect strategy ${run.label}`}
              >
                <span>{run.label}</span>
                <strong>{run.maxDrawdown == null ? "n/a" : formatPercent(run.maxDrawdown)}</strong>
                <em>
                  IR {run.informationRatio == null ? "n/a" : run.informationRatio.toFixed(2)} · {run.benchmark}
                </em>
              </button>
            );
          })}
        </div>
      </div>
    </motion.section>
  );
}
