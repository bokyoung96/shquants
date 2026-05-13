import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi, afterEach as afterEachHook } from "vitest";

const { fetchRuns, fetchDashboard, fetchSession } = vi.hoisted(() => ({
  fetchRuns: vi.fn(),
  fetchDashboard: vi.fn(),
  fetchSession: vi.fn(),
}));

vi.mock("../lib/api", () => ({
  fetchRuns,
  fetchDashboard,
  fetchSession,
}));

vi.mock("echarts-for-react", () => ({
  default: () => <div data-testid="chart" />,
}));

beforeEach(() => {
  fetchRuns.mockReset();
  fetchDashboard.mockReset();
  fetchSession.mockReset();
  fetchSession.mockResolvedValue({ defaultSelectedRunIds: ["trend_run", "trend_run"] });
  fetchRuns.mockResolvedValue([
    {
      run_id: "trend_run",
      label: "Trend Rank",
      strategy: "trend_rank",
      summary: { finalEquity: 100, avgTurnover: 0.1 },
    },
    {
      run_id: "trend_run",
      label: "Trend Rank",
      strategy: "trend_rank",
      summary: { finalEquity: 100, avgTurnover: 0.1 },
    },
    {
      run_id: "trend_variant_run",
      label: "Trend Rank Variant",
      strategy: "trend_rank",
      summary: { finalEquity: 105, avgTurnover: 0.2 },
    },
  ]);
  fetchDashboard.mockResolvedValue({
    mode: "single",
    selectedRunIds: ["trend_run"],
    availableRuns: [
      {
        run_id: "trend_run",
        label: "Trend Rank",
        strategy: "trend_rank",
        summary: { finalEquity: 100, avgTurnover: 0.1 },
      },
      {
        run_id: "trend_variant_run",
        label: "Trend Rank Variant",
        strategy: "trend_rank",
        summary: { finalEquity: 105, avgTurnover: 0.2 },
      },
    ],
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
        strategies: [],
      },
      asOfDate: "2025-12-31",
    },
    metrics: {
      trend_run: {
        label: "Trend Rank",
        cumulativeReturn: 0.12,
        cagr: 0.12,
        annualVolatility: 0.15,
        sharpe: 1.1,
        sortino: 1.3,
        calmar: 1.5,
        maxDrawdown: -0.08,
        avgTurnover: 0.1,
        finalEquity: 112,
        alpha: 0.01,
        beta: 0.9,
        trackingError: 0.04,
        informationRatio: 0.2,
      },
    },
    context: {
      trend_run: {
        label: "Trend Rank",
        strategy: "trend_rank",
        benchmark: { code: "KOSPI200", name: "KOSPI200 benchmark" },
        startDate: "2020-01-01",
        endDate: "2020-12-31",
        asOfDate: "2020-12-31",
      },
    },
    performance: {
      series: [
        {
          runId: "trend_run",
          label: "Trend Rank",
          points: [
            { date: "2025-01-01", value: 100 },
            { date: "2025-01-02", value: 101 },
          ],
        },
      ],
      benchmark: [
        { date: "2025-01-01", value: 100 },
        { date: "2025-01-02", value: 101 },
      ],
      benchmarks: [
        {
          runId: "trend_run",
          label: "KOSPI200 benchmark",
          points: [
            { date: "2025-01-01", value: 100 },
            { date: "2025-01-02", value: 101 },
          ],
        },
      ],
      drawdowns: [],
    },
    rolling: { rollingSharpe: [], rollingBeta: [], rollingCorrelation: [] },
    exposure: {
      holdingsCount: [],
      latestHoldings: {},
      latestHoldingsWinners: {},
      latestHoldingsLosers: {},
      sectorWeights: {},
    },
    research: {
      focus: { kind: "all-selected", label: "All Selected", value: null },
      sectorContributionMethod: "weighted-asset-return-attribution",
      monthlyHeatmap: {},
      returnDistribution: {},
      monthlyReturnDistribution: {},
      yearlyExcessReturns: {},
      sectorContributionSeries: {},
      sectorWeightSeries: {},
      drawdownEpisodes: {},
    },
  });
});

import { App } from "../app/App";

afterEachHook(() => {
  cleanup();
});

async function selectorScope() {
  const selector = (await screen.findByText("Select saved runs")).closest("section");
  if (!selector) {
    throw new Error("run selector not found");
  }

  return within(selector);
}

describe("Run selection", () => {
  it("suppresses duplicate run ids from the interactive selector and bootstrap request", async () => {
    render(<App />);

    const selector = await selectorScope();
    expect(selector.getAllByRole("button", { name: /^Trend Rank\s+trend_rank$/i })).toHaveLength(1);
    expect(selector.getAllByRole("button", { name: /Trend Rank Variant/i })).toHaveLength(1);
    await waitFor(() => expect(fetchDashboard).toHaveBeenCalledWith(["trend_run"]));
  });

  it("allows selecting and deselecting runs without introducing duplicate selected ids", async () => {
    const user = userEvent.setup();

    render(<App />);

    expect(await screen.findAllByText("1 selected")).toHaveLength(1);
    let selector = await selectorScope();
    await user.click(selector.getByRole("button", { name: /Trend Rank Variant/i }));

    expect(screen.getAllByText("2 selected")).toHaveLength(1);
    expect(fetchDashboard).toHaveBeenNthCalledWith(1, ["trend_run"]);
    expect(fetchDashboard).toHaveBeenNthCalledWith(2, ["trend_run", "trend_variant_run"]);

    selector = await selectorScope();
    await user.click(selector.getByRole("button", { name: /^Trend Rank\s+trend_rank$/i }));

    expect(screen.getAllByText("1 selected")).toHaveLength(1);
    expect(fetchDashboard).toHaveBeenNthCalledWith(3, ["trend_variant_run"]);

    selector = await selectorScope();
    await user.click(selector.getByRole("button", { name: /Trend Rank Variant/i }));

    expect(screen.getAllByText("0 selected")).toHaveLength(1);
    expect(fetchDashboard).toHaveBeenCalledTimes(3);
  });
});
