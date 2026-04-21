import EChartsReact from "echarts-for-react";
import { motion } from "framer-motion";

import { formatPercent } from "../lib/format";
import type { DashboardPayload, HoldingPerformance, ResearchFocus } from "../lib/types";

const MAX_VISIBLE_HOLDINGS = 5;

type ExposureBandProps = {
  dashboard: DashboardPayload;
  focus: ResearchFocus;
  onFocusChange: (focus: ResearchFocus) => void;
};

function resolveRunIds(dashboard: DashboardPayload, focus: ResearchFocus) {
  if (focus.kind === "strategy" && dashboard.selectedRunIds.includes(focus.runId)) {
    return [focus.runId];
  }

  return dashboard.selectedRunIds;
}

function buildSectorSnapshotOption(
  label: string,
  sectors: Array<{
    name: string;
    value: number;
  }>,
) {
  return {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item" as const,
      backgroundColor: "rgba(15, 18, 21, 0.96)",
      borderColor: "rgba(240, 164, 75, 0.22)",
      textStyle: { color: "#f7f0e7" },
      formatter: (params: { name: string; value: number }) => `${params.name}: ${formatPercent(params.value, 2)}`,
    },
    legend: {
      bottom: 0,
      textStyle: { color: "#bdaea1", fontFamily: "inherit" },
    },
    series: [
      {
        name: `${label} sector snapshot`,
        type: "pie" as const,
        radius: ["50%", "72%"],
        center: ["50%", "44%"],
        avoidLabelOverlap: true,
        label: { color: "#f7f0e7", formatter: "{b}" },
        labelLine: { lineStyle: { color: "rgba(247, 240, 231, 0.22)" } },
        itemStyle: { borderColor: "#12161a", borderWidth: 2 },
        data: sectors,
      },
    ],
  };
}

function renderHoldingRows(holdings: HoldingPerformance[]) {
  return holdings.length > 0 ? (
    holdings.map((holding) => (
      <div key={holding.symbol} className="detail-list-row">
        <strong>{holding.symbol}</strong>
        <span>{formatPercent(holding.absWeight)}</span>
        <span>{formatPercent(holding.returnSinceLatestRebalance, 2)}</span>
      </div>
    ))
  ) : (
    <div className="detail-list-row detail-list-row--empty">
      <strong>No holdings</strong>
      <span>latest snapshot missing</span>
    </div>
  );
}

export function ExposureBand({ dashboard, focus, onFocusChange }: ExposureBandProps) {
  const runIds = resolveRunIds(dashboard, focus);

  return (
    <motion.section
      className="detail-section exposure-band"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      aria-label="Exposure band"
    >
      <div className="detail-section-copy">
        <p className="section-label">Exposure band</p>
        <h2>Latest holdings and sector context</h2>
        <p className="workspace-summary">
          Sector rows act as in-page drill-down controls. Holdings stay pinned to the latest available snapshot.
        </p>
      </div>

      {focus.kind === "sector" ? (
        <div className="focus-banner focus-banner--inline">
          <span className="section-label">Sector drill-down</span>
          <strong>{focus.sectorName}</strong>
          <button type="button" className="workspace-inline-action" onClick={() => onFocusChange({ kind: "all-selected" })}>
            Show all selected
          </button>
        </div>
      ) : null}

      <div className="detail-run-list">
        {runIds.map((runId) => {
          const run = dashboard.availableRuns.find((entry) => entry.run_id === runId);
          const context = dashboard.context[runId];
          const holdings = (dashboard.exposure.latestHoldings[runId] ?? []).slice(0, MAX_VISIBLE_HOLDINGS);
          const sectorWeights = dashboard.exposure.sectorWeights[runId] ?? [];
          const visibleSectors =
            focus.kind === "sector"
              ? sectorWeights.filter((sector) => sector.name === focus.sectorName)
              : sectorWeights;

          return (
            <div key={runId} className="detail-run-block">
              <div className="detail-run-head">
                <strong>{context?.label ?? run?.label ?? runId}</strong>
                <span>{context?.strategy ?? run?.strategy ?? "strategy"}</span>
              </div>

              <div className="detail-subgrid">
                <div className="detail-subsection">
                  <div className="detail-subsection-head">
                    <span>Latest holdings</span>
                    <span>{holdings.length} lines</span>
                  </div>
                  <div className="detail-column-labels">
                    <span>Symbol</span>
                    <span>Target weight</span>
                    <span>Absolute weight</span>
                  </div>
                  <div className="detail-list">
                    {holdings.length > 0 ? (
                      holdings.map((holding) => (
                        <div key={holding.symbol} className="detail-list-row">
                          <strong>{holding.symbol}</strong>
                          <span>{formatPercent(holding.targetWeight)}</span>
                          <span>{formatPercent(holding.absWeight)}</span>
                        </div>
                      ))
                    ) : (
                      <div className="detail-list-row detail-list-row--empty">
                        <strong>No holdings</strong>
                        <span>latest snapshot missing</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="detail-subsection">
                  <div className="detail-subsection-head">
                    <span>Sector weights</span>
                    <span>{visibleSectors.length} sectors</span>
                  </div>
                  <div className="detail-column-labels detail-column-labels--sectors">
                    <span>Sector</span>
                    <span>Weight</span>
                  </div>
                  <div className="detail-list">
                    {visibleSectors.length > 0 ? (
                      visibleSectors.map((sector) => {
                        const isFocused = focus.kind === "sector" && focus.sectorName === sector.name;

                        return (
                          <button
                            key={sector.name}
                            type="button"
                            className={`detail-list-row detail-list-row--button ${isFocused ? "is-focused" : ""}`}
                            onClick={() => onFocusChange({ kind: "sector", sectorName: sector.name })}
                            aria-label={`Focus sector ${sector.name}`}
                          >
                            <strong>{sector.name}</strong>
                            <span>{formatPercent(sector.value)}</span>
                          </button>
                        );
                      })
                    ) : (
                      <div className="detail-list-row detail-list-row--empty">
                        <strong>No sectors</strong>
                        <span>latest snapshot missing</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="detail-panel-grid">
        <section className="detail-run-block" aria-label="Latest holdings winners">
          <div className="detail-subsection-head">
            <span>Latest holdings winners</span>
            <span>Top performers since rebalance</span>
          </div>
          <div className="detail-run-list">
            {runIds.map((runId) => {
              const context = dashboard.context[runId];
              const winners = (dashboard.exposure.latestHoldingsWinners[runId] ?? []).slice(0, MAX_VISIBLE_HOLDINGS);

              return (
                <div key={`winners-${runId}`} className="detail-run-block detail-run-block--nested">
                  <div className="detail-run-head">
                    <strong>{context?.label ?? runId}</strong>
                    <span>{winners.length} lines</span>
                  </div>
                  <div className="detail-column-labels">
                    <span>Symbol</span>
                    <span>Weight</span>
                    <span>Return</span>
                  </div>
                  <div className="detail-list">{renderHoldingRows(winners)}</div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="detail-run-block" aria-label="Latest holdings losers">
          <div className="detail-subsection-head">
            <span>Latest holdings losers</span>
            <span>Lagging since rebalance</span>
          </div>
          <div className="detail-run-list">
            {runIds.map((runId) => {
              const context = dashboard.context[runId];
              const losers = (dashboard.exposure.latestHoldingsLosers[runId] ?? []).slice(0, MAX_VISIBLE_HOLDINGS);

              return (
                <div key={`losers-${runId}`} className="detail-run-block detail-run-block--nested">
                  <div className="detail-run-head">
                    <strong>{context?.label ?? runId}</strong>
                    <span>{losers.length} lines</span>
                  </div>
                  <div className="detail-column-labels">
                    <span>Symbol</span>
                    <span>Weight</span>
                    <span>Return</span>
                  </div>
                  <div className="detail-list">{renderHoldingRows(losers)}</div>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      <section className="detail-run-block" aria-label="Latest sector snapshot">
        <div className="detail-subsection-head">
          <span>Latest sector snapshot</span>
          <span>Latest allocation mix by selected strategy</span>
        </div>
        <div className="detail-panel-grid">
          {runIds.map((runId) => {
            const run = dashboard.availableRuns.find((entry) => entry.run_id === runId);
            const context = dashboard.context[runId];
            const sectors = dashboard.exposure.sectorWeights[runId] ?? [];

            return (
              <div key={`sector-snapshot-${runId}`} className="detail-run-block detail-run-block--nested">
                <div className="detail-run-head">
                  <strong>{context?.label ?? run?.label ?? runId}</strong>
                  <span>{sectors.length} sectors</span>
                </div>
                {sectors.length > 0 ? (
                  <>
                    <div className="detail-chart-shell">
                      <EChartsReact
                        option={buildSectorSnapshotOption(context?.label ?? run?.label ?? runId, sectors)}
                        style={{ height: 260, width: "100%" }}
                      />
                    </div>
                    <div className="detail-note">
                      <strong>{sectors[0]?.name ?? "Top sector"}</strong>
                      <p>{sectors[0] ? `${formatPercent(sectors[0].value)} of latest snapshot.` : "No sector snapshot available."}</p>
                    </div>
                  </>
                ) : (
                  <div className="detail-note">
                    <strong>No sectors</strong>
                    <p>Latest sector snapshot is unavailable.</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </motion.section>
  );
}
