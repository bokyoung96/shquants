import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DashboardPayload } from "../lib/types";

const { fetchRuns, fetchDashboard, fetchSession } = vi.hoisted(() => ({
  fetchRuns: vi.fn(),
  fetchDashboard: vi.fn(),
  fetchSession: vi.fn(),
}));

const chartOptions: unknown[] = [];

vi.mock("../lib/api", () => ({
  fetchRuns,
  fetchDashboard,
  fetchSession,
}));

vi.mock("echarts-for-react", () => ({
  default: (props: { option?: unknown }) => {
    chartOptions.push(props.option);

    return <div data-testid="chart" />;
  },
}));

import { App } from "../app/App";

const RUNS = [
  {
    run_id: "trend_run",
    label: "Trend Rank",
    strategy: "trend_rank",
    summary: { finalEquity: 100000000, avgTurnover: 0.12 },
  },
  {
    run_id: "value_run",
    label: "Trend Rank Variant",
    strategy: "trend_rank",
    summary: { finalEquity: 105000000, avgTurnover: 0.2 },
  },
];

function createDashboard(mode: "single" | "multi", selectedRunIds: string[]): DashboardPayload {
  return {
    mode,
    selectedRunIds,
    availableRuns: RUNS,
      launch: {
        configuredStartDate: "2025-01-01",
        configuredEndDate: "2025-12-31",
        capital: 100000000,
        schedule: "monthly",
        fillMode: "next_open",
        fee: 0,
        sellTax: 0,
        slippage: 0,
        benchmark: {
        kind: "shared",
        shared: { code: "KOSPI200", name: "KOSPI200 benchmark" },
        strategies: [
          {
            strategy: "trend_rank",
            label: "Trend Rank",
            benchmark: { code: "KOSPI200", name: "KOSPI200 benchmark" },
          },
          {
            strategy: "trend_rank",
            label: "Trend Rank Variant",
            benchmark: { code: "SPX", name: "S&P 500 benchmark" },
          },
        ],
      },
      asOfDate: "2025-12-31",
    },
    metrics: Object.fromEntries(
      selectedRunIds.map((runId) => [
        runId,
        runId === "trend_run"
          ? {
              label: "Trend Rank",
              cumulativeReturn: 0.16,
              cagr: 0.13,
              annualVolatility: 0.18,
              sharpe: 1.05,
              sortino: 1.33,
              calmar: 1.08,
              maxDrawdown: -0.12,
              avgTurnover: 0.12,
              finalEquity: 108000000,
              alpha: 0.02,
              beta: 0.94,
              trackingError: 0.06,
              informationRatio: 0.31,
            }
          : {
              label: "Trend Rank Variant",
              cumulativeReturn: 0.1,
              cagr: 0.09,
              annualVolatility: 0.12,
              sharpe: 0.82,
              sortino: 1.06,
              calmar: 1.5,
              maxDrawdown: -0.06,
              avgTurnover: 0.2,
              finalEquity: 103000000,
              alpha: -0.01,
              beta: 0.71,
              trackingError: 0.05,
              informationRatio: -0.2,
            },
      ]),
    ),
    context: {
      trend_run: {
        label: "Trend Rank",
        strategy: "trend_rank",
        benchmark: { code: "KOSPI200", name: "KOSPI200 benchmark" },
        startDate: "2025-01-01",
        endDate: "2025-12-31",
        asOfDate: "2025-12-31",
      },
      value_run: {
        label: "Trend Rank Variant",
        strategy: "trend_rank",
        benchmark: { code: "SPX", name: "S&P 500 benchmark" },
        startDate: "2025-01-01",
        endDate: "2025-12-31",
        asOfDate: "2025-12-31",
      },
    },
    performance: {
      series: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          points: [
            { date: "2025-01-01", value: 100000000 },
            { date: "2025-02-01", value: 103000000 },
            { date: "2025-03-01", value: 108000000 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant",
          points: [
            { date: "2025-01-01", value: 100000000 },
            { date: "2025-02-01", value: 101000000 },
            { date: "2025-03-01", value: 103000000 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
      benchmark:
        mode === "single"
          ? [
              { date: "2025-01-01", value: 100000000 },
              { date: "2025-02-01", value: 101500000 },
              { date: "2025-03-01", value: 102500000 },
            ]
          : null,
      benchmarks: [
        {
          runId: "trend_run",
          label: "KOSPI200 benchmark",
          points: [
            { date: "2025-01-01", value: 100000000 },
            { date: "2025-02-01", value: 101500000 },
            { date: "2025-03-01", value: 102500000 },
          ],
        },
        {
          runId: "value_run",
          label: "S&P 500 benchmark",
          points: [
            { date: "2025-01-01", value: 100000000 },
            { date: "2025-02-01", value: 102500000 },
            { date: "2025-03-01", value: 104000000 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
      drawdowns: [
        {
          runId: "trend_run",
          label: "Trend Rank drawdown",
          points: [
            { date: "2025-01-01", value: 0 },
            { date: "2025-02-01", value: -0.05 },
            { date: "2025-03-01", value: -0.12 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant drawdown",
          points: [
            { date: "2025-01-01", value: 0 },
            { date: "2025-02-01", value: -0.02 },
            { date: "2025-03-01", value: -0.06 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
    },
    rolling: {
      rollingSharpe: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          points: [
            { date: "2025-01-01", value: 0.8 },
            { date: "2025-02-01", value: 1.05 },
            { date: "2025-03-01", value: 1.14 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant",
          points: [
            { date: "2025-01-01", value: 0.52 },
            { date: "2025-02-01", value: 0.74 },
            { date: "2025-03-01", value: 0.81 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
      rollingBeta: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          points: [
            { date: "2025-01-01", value: 0.88 },
            { date: "2025-02-01", value: 0.94 },
            { date: "2025-03-01", value: 1.01 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant",
          points: [
            { date: "2025-01-01", value: 0.61 },
            { date: "2025-02-01", value: 0.69 },
            { date: "2025-03-01", value: 0.73 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
      rollingCorrelation: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          benchmark: { code: "KOSPI200", name: "KOSPI200 benchmark" },
          window: 252,
          points: [
            { date: "2025-01-01", value: 0.72 },
            { date: "2025-02-01", value: 0.8 },
            { date: "2025-03-01", value: 0.85 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant",
          benchmark: { code: "SPX", name: "S&P 500 benchmark" },
          window: 252,
          points: [
            { date: "2025-01-01", value: 0.58 },
            { date: "2025-02-01", value: 0.63 },
            { date: "2025-03-01", value: 0.67 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
    },
    exposure: {
      holdingsCount: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          points: [
            { date: "2025-01-01", value: 18 },
            { date: "2025-03-01", value: 22 },
          ],
        },
        {
          runId: "value_run",
          label: "Trend Rank Variant",
          points: [
            { date: "2025-01-01", value: 24 },
            { date: "2025-03-01", value: 19 },
          ],
        },
      ].filter((entry) => selectedRunIds.includes(entry.runId)),
      latestHoldings: {
        trend_run: [
          { symbol: "AAPL", targetWeight: 0.34, absWeight: 0.34 },
          { symbol: "MSFT", targetWeight: 0.21, absWeight: 0.21 },
          { symbol: "NVDA", targetWeight: 0.18, absWeight: 0.18 },
        ],
        value_run: [
          { symbol: "XOM", targetWeight: 0.28, absWeight: 0.28 },
          { symbol: "JPM", targetWeight: 0.19, absWeight: 0.19 },
          { symbol: "UNP", targetWeight: 0.16, absWeight: 0.16 },
        ],
      },
      latestHoldingsWinners: {
        trend_run: [
          { symbol: "TSLA", targetWeight: 0.04, absWeight: 0.04, returnSinceLatestRebalance: 0.3 },
          { symbol: "GOOG", targetWeight: 0.06, absWeight: 0.06, returnSinceLatestRebalance: 0.15 },
          { symbol: "AMZN", targetWeight: 0.1, absWeight: 0.1, returnSinceLatestRebalance: 0.1 },
          { symbol: "NVDA", targetWeight: 0.15, absWeight: 0.15, returnSinceLatestRebalance: 0.05 },
          { symbol: "MSFT", targetWeight: 0.25, absWeight: 0.25, returnSinceLatestRebalance: 0.01 },
        ],
        value_run: [
          { symbol: "XOM", targetWeight: 0.28, absWeight: 0.28, returnSinceLatestRebalance: 0.12 },
          { symbol: "JPM", targetWeight: 0.19, absWeight: 0.19, returnSinceLatestRebalance: 0.09 },
          { symbol: "UNP", targetWeight: 0.16, absWeight: 0.16, returnSinceLatestRebalance: 0.04 },
          { symbol: "CVX", targetWeight: 0.12, absWeight: 0.12, returnSinceLatestRebalance: 0.03 },
          { symbol: "CAT", targetWeight: 0.08, absWeight: 0.08, returnSinceLatestRebalance: 0.01 },
        ],
      },
      latestHoldingsLosers: {
        trend_run: [
          { symbol: "AAPL", targetWeight: 0.34, absWeight: 0.34, returnSinceLatestRebalance: -0.04 },
          { symbol: "MSFT", targetWeight: 0.21, absWeight: 0.21, returnSinceLatestRebalance: 0.01 },
          { symbol: "NVDA", targetWeight: 0.18, absWeight: 0.18, returnSinceLatestRebalance: 0.05 },
          { symbol: "AMZN", targetWeight: 0.1, absWeight: 0.1, returnSinceLatestRebalance: 0.1 },
          { symbol: "GOOG", targetWeight: 0.06, absWeight: 0.06, returnSinceLatestRebalance: 0.15 },
        ],
        value_run: [
          { symbol: "XOM", targetWeight: 0.28, absWeight: 0.28, returnSinceLatestRebalance: -0.08 },
          { symbol: "JPM", targetWeight: 0.19, absWeight: 0.19, returnSinceLatestRebalance: -0.02 },
          { symbol: "UNP", targetWeight: 0.16, absWeight: 0.16, returnSinceLatestRebalance: 0.01 },
          { symbol: "CVX", targetWeight: 0.12, absWeight: 0.12, returnSinceLatestRebalance: 0.03 },
          { symbol: "CAT", targetWeight: 0.08, absWeight: 0.08, returnSinceLatestRebalance: 0.05 },
        ],
      },
      sectorWeights: {
        trend_run: [
          { name: "Tech", value: 0.62 },
          { name: "Financials", value: 0.2 },
          { name: "Industrials", value: 0.12 },
        ],
        value_run: [
          { name: "Energy", value: 0.28 },
          { name: "Financials", value: 0.26 },
          { name: "Industrials", value: 0.18 },
        ],
      },
    },
    research: {
      focus: { kind: "all-selected", label: "All Selected", value: null },
      sectorContributionMethod: "weighted-asset-return-attribution",
      monthlyHeatmap: {
        trend_run: [
          { year: 2025, month: 1, value: 0.03 },
          { year: 2025, month: 2, value: -0.02 },
          { year: 2025, month: 3, value: 0.05 },
        ],
        value_run: [
          { year: 2025, month: 1, value: 0.01 },
          { year: 2025, month: 2, value: 0.02 },
          { year: 2025, month: 3, value: 0.01 },
        ],
      },
      returnDistribution: {
        trend_run: [
          { start: -0.04, end: -0.02, count: 1, frequency: 0.2 },
          { start: -0.02, end: 0, count: 1, frequency: 0.2 },
          { start: 0, end: 0.02, count: 1, frequency: 0.2 },
          { start: 0.02, end: 0.04, count: 2, frequency: 0.4 },
        ],
        value_run: [
          { start: -0.03, end: -0.01, count: 1, frequency: 0.2 },
          { start: -0.01, end: 0.01, count: 2, frequency: 0.4 },
          { start: 0.01, end: 0.03, count: 2, frequency: 0.4 },
        ],
      },
      monthlyReturnDistribution: {
        trend_run: [
          { start: -0.06, end: -0.03, count: 1, frequency: 0.17 },
          { start: -0.03, end: 0, count: 1, frequency: 0.17 },
          { start: 0, end: 0.03, count: 2, frequency: 0.33 },
          { start: 0.03, end: 0.06, count: 2, frequency: 0.33 },
        ],
        value_run: [
          { start: -0.04, end: -0.02, count: 1, frequency: 0.17 },
          { start: -0.02, end: 0, count: 1, frequency: 0.17 },
          { start: 0, end: 0.02, count: 2, frequency: 0.33 },
          { start: 0.02, end: 0.04, count: 2, frequency: 0.33 },
        ],
      },
      yearlyExcessReturns: {
        trend_run: [{ date: "2025-12-31", value: 0.04 }],
        value_run: [{ date: "2025-12-31", value: -0.01 }],
      },
      sectorContributionSeries: {
        trend_run: [
          {
            name: "Tech",
            points: [
              { date: "2025-01-01", value: 0 },
              { date: "2025-02-01", value: 0.02 },
              { date: "2025-03-01", value: 0.05 },
            ],
          },
          {
            name: "Financials",
            points: [
              { date: "2025-01-01", value: 0 },
              { date: "2025-02-01", value: 0.01 },
              { date: "2025-03-01", value: 0.02 },
            ],
          },
        ],
        value_run: [
          {
            name: "Energy",
            points: [
              { date: "2025-01-01", value: 0 },
              { date: "2025-02-01", value: 0.01 },
              { date: "2025-03-01", value: 0.02 },
            ],
          },
          {
            name: "Industrials",
            points: [
              { date: "2025-01-01", value: 0 },
              { date: "2025-02-01", value: 0.01 },
              { date: "2025-03-01", value: 0.015 },
            ],
          },
        ],
      },
      sectorWeightSeries: {
        trend_run: [
          {
            name: "Tech",
            points: [
              { date: "2025-01-01", value: 0.55 },
              { date: "2025-02-01", value: 0.58 },
              { date: "2025-03-01", value: 0.62 },
            ],
          },
          {
            name: "Financials",
            points: [
              { date: "2025-01-01", value: 0.18 },
              { date: "2025-02-01", value: 0.19 },
              { date: "2025-03-01", value: 0.2 },
            ],
          },
        ],
        value_run: [
          {
            name: "Energy",
            points: [
              { date: "2025-01-01", value: -0.2 },
              { date: "2025-02-01", value: -0.24 },
              { date: "2025-03-01", value: -0.28 },
            ],
          },
          {
            name: "Industrials",
            points: [
              { date: "2025-01-01", value: 0.16 },
              { date: "2025-02-01", value: 0.17 },
              { date: "2025-03-01", value: 0.18 },
            ],
          },
        ],
      },
      drawdownEpisodes: {
        trend_run: [
          {
            peak: "2025-01-20",
            start: "2025-01-21",
            trough: "2025-02-18",
            end: "2025-03-10",
            drawdown: -0.12,
            durationDays: 48,
            timeToTroughDays: 28,
            recoveryDays: 20,
            recovered: true,
          },
        ],
        value_run: [
          {
            peak: "2025-02-05",
            start: "2025-02-06",
            trough: "2025-02-25",
            end: "2025-03-18",
            drawdown: -0.06,
            durationDays: 40,
            timeToTroughDays: 19,
            recoveryDays: 21,
            recovered: true,
          },
        ],
      },
    },
  };
}

function createDeferredPromise<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve;
    reject = innerReject;
  });

  return { promise, resolve, reject };
}

function findChartOption(names: string[]) {
  return chartOptions.find((option) => {
    const series = (option as { series?: Array<{ name?: string }> }).series ?? [];
    const seriesNames = series.map((entry) => entry.name);
    return names.every((name) => seriesNames.includes(name));
  });
}

function findDistributionChartOption() {
  return chartOptions.find((option) => {
    const xAxis = (option as { xAxis?: { type?: string } }).xAxis;
    const series = (option as { series?: Array<{ name?: string; type?: string }> }).series ?? [];
    const seriesNames = series.map((entry) => entry.name);
    return xAxis?.type === "category" && series.every((entry) => entry.type === "bar") && seriesNames.includes("Trend Rank");
  });
}

function findStrategySectorWeightOption() {
  return chartOptions.find((option) => {
    const yAxis = (option as { yAxis?: { min?: number; max?: number } }).yAxis;
    const series = (option as { series?: Array<{ type?: string; name?: string }> }).series ?? [];
    const names = series.map((entry) => entry.name);
    return yAxis?.min === 0 && yAxis?.max === 1 && series.every((entry) => entry.type === "line") && names.includes("Tech");
  });
}

function findResearchReturnDrawdownOption() {
  return chartOptions.find((option) => {
    const xAxis = (option as { xAxis?: unknown }).xAxis;
    const yAxis = (option as { yAxis?: unknown }).yAxis;
    return Array.isArray(xAxis) && Array.isArray(yAxis);
  });
}

type ChartLineSeries = {
  name?: string;
  type?: string;
  data?: Array<[string | number, number]>;
  tooltip?: { valueFormatter?: (value: number) => string };
};

function findRollingSharpeOption() {
  return chartOptions.find((option) => {
    const xAxis = (option as { xAxis?: { type?: string } }).xAxis;
    const series = (option as { series?: ChartLineSeries[] }).series ?? [];
    const momentumSeries = series.find((entry) => entry.name === "Trend Rank");
    const valueSeries = series.find((entry) => entry.name === "Trend Rank Variant");
    return (
      xAxis?.type === "time" &&
      series.every((entry) => entry.type === "line") &&
      momentumSeries?.data?.some((point) => point[1] === 1.05) &&
      momentumSeries?.data?.some((point) => point[1] === 1.14) &&
      valueSeries?.data?.some((point) => point[1] === 0.52) &&
      valueSeries?.data?.some((point) => point[1] === 0.81)
    );
  });
}

function findRollingCorrelationOption() {
  return chartOptions.find((option) => {
    const series = (option as { series?: ChartLineSeries[] }).series ?? [];
    const momentumSeries = series.find((entry) => entry.name === "Trend Rank");
    const valueSeries = series.find((entry) => entry.name === "Trend Rank Variant");
    return (
      series.length === 2 &&
      series.every((entry) => entry.type === "line") &&
      momentumSeries?.data?.some((point) => point[1] === 0.72) &&
      valueSeries?.data?.some((point) => point[1] === 0.58)
    );
  });
}

function findRollingBetaOption() {
  return chartOptions.find((option) => {
    const series = (option as { series?: ChartLineSeries[] }).series ?? [];
    const momentumSeries = series.find((entry) => entry.name === "Trend Rank");
    const valueSeries = series.find((entry) => entry.name === "Trend Rank Variant");
    return (
      series.length === 2 &&
      series.every((entry) => entry.type === "line") &&
      momentumSeries?.data?.some((point) => point[1] === 0.88) &&
      valueSeries?.data?.some((point) => point[1] === 0.61)
    );
  });
}

function findDailyDistributionOption() {
  return chartOptions.find((option) => {
    const xAxis = (option as { xAxis?: { type?: string; data?: string[] } }).xAxis;
    const series = (option as { series?: Array<{ name?: string; type?: string; data?: number[] }> }).series ?? [];
    const seriesNames = series.map((entry) => entry.name);
    const momentumSeries = series.find((entry) => entry.name === "Trend Rank");
    return (
      xAxis?.type === "category" &&
      xAxis.data?.includes("-4.0% to -2.0%") &&
      xAxis.data?.includes("2.0% to 4.0%") &&
      series.every((entry) => entry.type === "bar") &&
      seriesNames.includes("Trend Rank") &&
      momentumSeries?.data?.includes(0.2) &&
      momentumSeries?.data?.includes(0.4)
    );
  });
}

function findMonthlyDistributionOption() {
  return chartOptions.find((option) => {
    const xAxis = (option as { xAxis?: { type?: string; data?: string[] } }).xAxis;
    const series = (option as { series?: Array<{ name?: string; type?: string; data?: number[] }> }).series ?? [];
    const seriesNames = series.map((entry) => entry.name);
    const momentumSeries = series.find((entry) => entry.name === "Trend Rank");
    return (
      xAxis?.type === "category" &&
      xAxis.data?.includes("-6.0% to -3.0%") &&
      xAxis.data?.includes("3.0% to 6.0%") &&
      series.every((entry) => entry.type === "bar") &&
      seriesNames.includes("Trend Rank") &&
      momentumSeries?.data?.includes(0.17) &&
      momentumSeries?.data?.includes(0.33)
    );
  });
}

function findStrategySectorContributionOption() {
  return chartOptions.find((option) => {
    const series = (option as { series?: Array<{ name?: string; type?: string; data?: number[] }> }).series ?? [];
    const names = series.map((entry) => entry.name);
    return (
      series.every((entry) => entry.type === "line") &&
      names.includes("Tech") &&
      names.includes("Financials") &&
      series.some((entry) => entry.data?.includes(0.05))
    );
  });
}

function findFilteredStrategyContributionOption(seriesName: string) {
  return chartOptions.find((option) => {
    const series = (option as { series?: Array<{ name?: string; type?: string }> }).series ?? [];
    const names = series.map((entry) => entry.name);
    return series.every((entry) => entry.type === "line") && names.length > 0 && names.every((name) => name === seriesName);
  });
}

function findFilteredStrategyWeightOption(seriesName: string) {
  return chartOptions.find((option) => {
    const yAxis = (option as { yAxis?: { min?: number; max?: number } }).yAxis;
    const series = (option as { series?: Array<{ name?: string; type?: string }> }).series ?? [];
    const names = series.map((entry) => entry.name);
    return (
      yAxis?.min === 0 &&
      yAxis?.max === 1 &&
      series.every((entry) => entry.type === "line") &&
      names.length > 0 &&
      names.every((name) => name === seriesName)
    );
  });
}

async function selectorScope() {
  const selector = (await screen.findByText("Select saved runs")).closest("section");
  if (!selector) {
    throw new Error("run selector not found");
  }

  return within(selector);
}

describe("App", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    fetchRuns.mockReset();
    fetchDashboard.mockReset();
    fetchSession.mockReset();
    fetchSession.mockResolvedValue({ defaultSelectedRunIds: [] });
    chartOptions.length = 0;
  });

  it("renders the brand and selection heading", async () => {
    fetchRuns.mockResolvedValue([RUNS[0]]);
    fetchDashboard.mockResolvedValue(createDashboard("single", ["trend_run"]));

    render(<App />);

    expect(await screen.findByText("Dashboard")).toBeInTheDocument();
    expect(await screen.findByText("Live Performance")).toBeInTheDocument();
    expect(await screen.findByText("Select saved runs")).toBeInTheDocument();
    const selector = await selectorScope();
    expect(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toBeInTheDocument();
  });

  it("renders the research workspace and updates focus from strategy and sector clicks", async () => {
    const user = userEvent.setup();
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Research charts" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Daily return distribution" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Monthly heatmap" })).toBeInTheDocument();
    expect(screen.getByText("Focus: All selected")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Focus strategy Trend Rank" }));

    expect(screen.getByText((text) => text.startsWith("Focus: Strategy") && text.includes("Trend Rank"))).toBeInTheDocument();
    expect(screen.getByText("Sharpe")).toBeInTheDocument();

    const exposureBand = screen.getByRole("region", { name: "Exposure band" });
    await user.click(within(exposureBand).getByRole("button", { name: "Focus sector Tech" }));

    expect(screen.getByText((text) => text.startsWith("Focus: Sector") && text.includes("Tech"))).toBeInTheDocument();
    expect(screen.getByText("Hit Rate")).toBeInTheDocument();
    expect(screen.getByText("Profit / Risk")).toBeInTheDocument();
  });

  it("shows a simple empty-state message when return distribution data is unavailable", async () => {
    fetchRuns.mockResolvedValue([RUNS[0]]);
    const dashboard = createDashboard("single", ["trend_run"]);
    dashboard.research.returnDistribution.trend_run = [];
    fetchDashboard.mockResolvedValue(dashboard);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Research charts" })).toBeInTheDocument();
    expect(screen.getByText("Returns, risk, and sector trends for the selected strategies.")).toBeInTheDocument();
    expect(screen.getByText("No return distribution data.")).toBeInTheDocument();
  });

  it("renders launch metadata in the comparison plane", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    const comparisonPlane = (await screen.findByText("Comparison plane")).closest("section");
    if (!comparisonPlane) {
      throw new Error("comparison plane not found");
    }

    expect(within(comparisonPlane).getByText("Launch metadata")).toBeInTheDocument();
    expect(within(comparisonPlane).getByText("Configured start")).toBeInTheDocument();
    expect(within(comparisonPlane).getByText("Configured end")).toBeInTheDocument();
    expect(within(comparisonPlane).getByText("Benchmark")).toBeInTheDocument();
    expect(within(comparisonPlane).getByText("Costs")).toBeInTheDocument();
  });

  it("renders rolling risk diagnostics for Sharpe, correlation, and beta", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Research charts" })).toBeInTheDocument();
    const rollingSharpeOption = findRollingSharpeOption() as {
      series: Array<ChartLineSeries>;
    };
    const momentumSharpeSeries = rollingSharpeOption.series.find((series) => series.name === "Trend Rank");
    expect(momentumSharpeSeries?.tooltip?.valueFormatter?.(1.14)).toBe("1.14");
    expect(findRollingCorrelationOption()).toMatchObject({
      series: [
        expect.objectContaining({
          name: "Trend Rank",
          type: "line",
          data: expect.arrayContaining([expect.arrayContaining([expect.any(String), 0.72])]),
        }),
        expect.objectContaining({
          name: "Trend Rank Variant",
          type: "line",
          data: expect.arrayContaining([expect.arrayContaining([expect.any(String), 0.58])]),
        }),
      ],
    });
    expect(findRollingBetaOption()).toMatchObject({
      series: [
        expect.objectContaining({
          name: "Trend Rank",
          type: "line",
          data: expect.arrayContaining([expect.arrayContaining([expect.any(String), 0.88])]),
        }),
        expect.objectContaining({
          name: "Trend Rank Variant",
          type: "line",
          data: expect.arrayContaining([expect.arrayContaining([expect.any(String), 0.61])]),
        }),
      ],
    });
  });

  it("shows daily and monthly return distributions separately", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Research charts" })).toBeInTheDocument();
    expect(findDailyDistributionOption()).toMatchObject({
      series: [
        expect.objectContaining({
          name: "Trend Rank",
          type: "bar",
          data: expect.arrayContaining([expect.any(Number)]),
        }),
        expect.objectContaining({
          name: "Trend Rank Variant",
          type: "bar",
        }),
      ],
    });
    expect(findMonthlyDistributionOption()).toMatchObject({
      series: [
        expect.objectContaining({
          name: "Trend Rank",
          type: "bar",
          data: expect.arrayContaining([expect.any(Number)]),
        }),
        expect.objectContaining({
          name: "Trend Rank Variant",
          type: "bar",
        }),
      ],
    });
  });

  it("renders latest-holdings winners and losers panels", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    const exposureBand = await screen.findByRole("region", { name: "Exposure band" });
    const exposureHeading = within(exposureBand).getByRole("heading", { name: "Latest holdings and sector context" });
    const exposurePanel = exposureHeading.closest("section");
    if (!exposurePanel) {
      throw new Error("exposure panel not found");
    }

    const winnersPanel = within(exposurePanel).getByRole("region", { name: /Latest holdings winners/i });
    const losersPanel = within(exposurePanel).getByRole("region", { name: /Latest holdings losers/i });

    expect(within(winnersPanel).getByText("TSLA")).toBeInTheDocument();
    expect(within(losersPanel).getByText("AAPL")).toBeInTheDocument();
  });

  it("renders return distribution as a numeric distribution curve", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    expect(findDistributionChartOption()).toMatchObject({
      xAxis: expect.objectContaining({ type: "category" }),
      series: [
        expect.objectContaining({ name: "Trend Rank", type: "bar" }),
        expect.objectContaining({ name: "Trend Rank Variant", type: "bar" }),
      ],
    });
  });

  it("formats return distribution tooltip values as percentages", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    const option = findDistributionChartOption() as {
      tooltip?: { formatter?: (params: Array<{ axisValue: string; seriesName: string; data: number }>) => string };
    };
    const output = option.tooltip?.formatter?.([
      { axisValue: "1.0% to 3.0%", seriesName: "Trend Rank", data: 0.4 },
    ]);

    expect(output).toContain("1.0% to 3.0%");
    expect(output).toContain("40.00%");
  });

  it("filters sector charts with explicit sector selection controls", async () => {
    const user = userEvent.setup();
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });
    expect(screen.getByRole("button", { name: "Show all sectors" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Toggle sector Tech" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Toggle sector Energy" })).toBeInTheDocument();

    chartOptions.length = 0;
    await user.click(screen.getByRole("button", { name: "Toggle sector Energy" }));

    const contributionOption = findFilteredStrategyContributionOption("Energy");
    const weightOption = findFilteredStrategyWeightOption("Energy");
    expect(contributionOption).toMatchObject({
      series: [expect.objectContaining({ name: "Energy" })],
    });
    expect(weightOption).toMatchObject({
      series: [expect.objectContaining({ name: "Energy" })],
    });
    expect(JSON.stringify(contributionOption)).not.toContain("Tech");
    expect(JSON.stringify(weightOption)).not.toContain("Tech");
    expect(JSON.stringify(weightOption)).not.toContain("Industrials");
  });

  it("lets sector drill-down override a previously selected manual sector filter", async () => {
    const user = userEvent.setup();
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });
    await user.click(screen.getByRole("button", { name: "Toggle sector Energy" }));

    chartOptions.length = 0;
    const exposureBand = screen.getByRole("region", { name: "Exposure band" });
    await user.click(within(exposureBand).getByRole("button", { name: "Focus sector Tech" }));

    const contributionOption = findFilteredStrategyContributionOption("Tech");
    const weightOption = findFilteredStrategyWeightOption("Tech");
    expect(screen.getByText((text) => text.startsWith("Focus: Sector") && text.includes("Tech"))).toBeInTheDocument();
    expect(JSON.stringify(contributionOption)).toContain("Tech");
    expect(JSON.stringify(weightOption)).toContain("Tech");
    expect(JSON.stringify(contributionOption)).not.toContain("Energy");
    expect(JSON.stringify(weightOption)).not.toContain("Energy");
  });

  it("renders strategy sector weights as normalized trend lines", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    expect(findStrategySectorWeightOption()).toMatchObject({
      yAxis: expect.objectContaining({
        min: 0,
        max: 1,
      }),
      series: expect.arrayContaining([
        expect.objectContaining({ type: "line", name: "Tech" }),
        expect.objectContaining({ type: "line", name: "Financials" }),
      ]),
    });
  });

  it("normalizes sector weights to percentages on the trend chart", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    const option = findStrategySectorWeightOption() as {
      yAxis?: { min?: number; max?: number; axisLabel?: { formatter?: (value: number) => string } };
    };

    expect(option.yAxis?.min).toBe(0);
    expect(option.yAxis?.max).toBe(1);
    expect(option.yAxis?.axisLabel?.formatter?.(0.62)).toBe("62%");
  });

  it("formats research return and drawdown tooltips with money and percentages", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    const option = findResearchReturnDrawdownOption() as {
      series: Array<{ name: string; tooltip?: { valueFormatter?: (value: number) => string } }>;
    };
    const momentumSeries = option.series.find((series) => series.name === "Trend Rank");
    const drawdownSeries = option.series.find((series) => series.name === "Trend Rank drawdown");

    expect(momentumSeries?.tooltip?.valueFormatter?.(108000000)).toBe("$108.0M");
    expect(drawdownSeries?.tooltip?.valueFormatter?.(-0.12)).toBe("-12.00%");
  });

  it("formats rolling and sector contribution line tooltips as short percentages", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    const rollingOption = findRollingSharpeOption() as {
      series: Array<{ name: string; tooltip?: { valueFormatter?: (value: number) => string } }>;
    };
    const sectorContributionOption = findStrategySectorContributionOption() as {
      series: Array<{ name: string; tooltip?: { valueFormatter?: (value: number) => string } }>;
    };

    expect(rollingOption.series[0]?.tooltip?.valueFormatter?.(1.14)).toBe("1.14");
    expect(sectorContributionOption.series[0]?.tooltip?.valueFormatter?.(0.05)).toBe("5.00%");
  });

  it("does not cumulatively sum sector contribution a second time in the strategy trend chart", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    const option = findStrategySectorContributionOption() as {
      series: Array<{ name?: string; data?: number[] }>;
    };
    const techSeries = option.series.find((series) => series.name === "Tech");

    expect(techSeries?.data).toEqual([0, 0.02, 0.05]);
  });

  it("plots each selected strategy beside its corresponding benchmark overlay", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    await screen.findByRole("heading", { name: "Research charts" });

    expect(
      findChartOption(["Trend Rank", "KOSPI200 benchmark", "Trend Rank Variant", "S&P 500 benchmark"]),
    ).toMatchObject({
      series: [
        expect.objectContaining({ name: "Trend Rank" }),
        expect.objectContaining({ name: "KOSPI200 benchmark" }),
        expect.objectContaining({ name: "Trend Rank Variant" }),
        expect.objectContaining({ name: "S&P 500 benchmark" }),
      ],
    });
  });

  it("deduplicates bootstrap-selected ids before requesting the dashboard", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchSession.mockResolvedValue({
      defaultSelectedRunIds: ["value_run", "value_run", "trend_run", "value_run"],
    });
    fetchDashboard.mockResolvedValue(createDashboard("multi", ["trend_run", "value_run"]));

    render(<App />);

    const selector = await selectorScope();
    expect(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toHaveAttribute("aria-pressed", "true");
    expect(selector.getByRole("button", { name: /Trend Rank Variant/i })).toHaveAttribute("aria-pressed", "true");
    expect(fetchDashboard).toHaveBeenCalledWith(["trend_run", "value_run"]);
  });

  it("renders a failure message when saved runs fail to load", async () => {
    fetchRuns.mockRejectedValue(new Error("Failed to load saved runs."));

    render(<App />);

    expect(await screen.findByText("Failed to load saved runs.")).toBeInTheDocument();
  });

  it("does not show the empty-state copy while saved runs are still loading", () => {
    fetchRuns.mockReturnValue(new Promise(() => undefined));

    render(<App />);

    expect(screen.queryByRole("heading", { name: /No saved runs/i })).not.toBeInTheDocument();
  });

  it("clears dashboard errors when all runs are deselected", async () => {
    const user = userEvent.setup();
    fetchRuns.mockResolvedValue([RUNS[0]]);
    fetchDashboard.mockRejectedValue(new Error("Dashboard unavailable"));

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Dashboard unavailable" })).toBeInTheDocument();
    expect(fetchDashboard).toHaveBeenCalledWith(["trend_run"]);

    const selector = await selectorScope();
    await user.click(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i }));

    expect(screen.queryByRole("heading", { name: "Dashboard unavailable" })).not.toBeInTheDocument();
    expect(screen.getByText("0 selected")).toBeInTheDocument();
  });

  it("removes stale dashboard content when a later dashboard request fails", async () => {
    const user = userEvent.setup();
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard
      .mockResolvedValueOnce(createDashboard("single", ["trend_run"]))
      .mockRejectedValueOnce(new Error("Failed to load dashboard."));

    render(<App />);

    expect(await screen.findByText("Single strategy view")).toBeInTheDocument();

    const selector = await selectorScope();
    await user.click(selector.getByRole("button", { name: /Trend Rank Variant/i }));

    expect(await screen.findByText("Failed to load dashboard.")).toBeInTheDocument();
    expect(screen.queryByText("Single strategy view")).not.toBeInTheDocument();
    expect(fetchDashboard).toHaveBeenNthCalledWith(1, ["trend_run"]);
    expect(fetchDashboard).toHaveBeenNthCalledWith(2, ["trend_run", "value_run"]);
  });

  it("renders dashboard errors without clearing the saved-run selector", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchDashboard.mockRejectedValue(new Error("Failed to load dashboard."));

    render(<App />);

    expect(await screen.findByText("Failed to load dashboard.")).toBeInTheDocument();
    const selector = await selectorScope();
    expect(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toBeInTheDocument();
    expect(screen.getByText("1 selected")).toBeInTheDocument();
  });

  it("falls back to the newest available run when bootstrap ids are stale", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchSession.mockResolvedValue({ defaultSelectedRunIds: ["missing_run"] });
    fetchDashboard.mockResolvedValue(createDashboard("single", ["trend_run"]));

    render(<App />);

    const selector = await selectorScope();
    expect(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toHaveAttribute("aria-pressed", "true");
    expect(fetchDashboard).toHaveBeenCalledWith(["trend_run"]);
  });

  it("falls back to the newest available run when the session bootstrap request fails", async () => {
    fetchRuns.mockResolvedValue(RUNS);
    fetchSession.mockRejectedValue(new Error("session unavailable"));
    fetchDashboard.mockResolvedValue(createDashboard("single", ["trend_run"]));

    render(<App />);

    const selector = await selectorScope();
    expect(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toHaveAttribute("aria-pressed", "true");
    expect(fetchDashboard).toHaveBeenCalledWith(["trend_run"]);
  });

  it("does not show the empty state after runs load until the request resolves with zero items", async () => {
    const deferredRuns = createDeferredPromise<(typeof RUNS)[number][]>();
    fetchRuns.mockReturnValue(deferredRuns.promise);

    render(<App />);

    expect(screen.queryByRole("heading", { name: /No saved runs/i })).not.toBeInTheDocument();

    deferredRuns.resolve([]);

    expect(await screen.findByRole("heading", { name: /No saved runs/i })).toBeInTheDocument();
  });
});
