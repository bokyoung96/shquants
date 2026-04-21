import { useEffect, useState } from "react";

import { DiagnosticStrip } from "../components/DiagnosticStrip";
import { EmptyState } from "../components/EmptyState";
import { ErrorState } from "../components/ErrorState";
import { ExposureBand } from "../components/ExposureBand";
import { PerformanceStrip } from "../components/PerformanceStrip";
import { ResearchWorkspace } from "../components/ResearchWorkspace";
import { RunSelector } from "../components/RunSelector";
import { TopRail } from "../components/TopRail";
import { fetchDashboard, fetchRuns, fetchSession } from "../lib/api";
import type { DashboardPayload, ResearchFocus, RunOption, SessionBootstrap } from "../lib/types";

function uniqueRunOptions(runs: RunOption[]) {
  const seen = new Set<string>();
  return runs.filter((run) => {
    if (seen.has(run.run_id)) {
      return false;
    }

    seen.add(run.run_id);
    return true;
  });
}

function uniqueRunIds(runIds: string[]) {
  const seen = new Set<string>();
  return runIds.filter((runId) => {
    if (seen.has(runId)) {
      return false;
    }

    seen.add(runId);
    return true;
  });
}

function orderSelectedRunIds(runIds: string[], runs: RunOption[]) {
  const uniqueIds = new Set(uniqueRunIds(runIds));
  return runs.filter((run) => uniqueIds.has(run.run_id)).map((run) => run.run_id);
}

function resolveInitialRunIds(runs: RunOption[], bootstrap: SessionBootstrap) {
  const availableRunIds = new Set(runs.map((run) => run.run_id));
  const validBootstrapIds = uniqueRunIds(bootstrap.defaultSelectedRunIds).filter((runId) => availableRunIds.has(runId));

  if (validBootstrapIds.length > 0) {
    return orderSelectedRunIds(validBootstrapIds, runs);
  }

  return runs[0] ? [runs[0].run_id] : [];
}

function normalizeDashboardSelection(dashboard: DashboardPayload, runs: RunOption[]) {
  const normalizedRunIds = orderSelectedRunIds(
    dashboard.selectedRunIds.length > 0 ? dashboard.selectedRunIds : dashboard.availableRuns.map((run) => run.run_id),
    runs,
  );

  return {
    ...dashboard,
    selectedRunIds: normalizedRunIds,
    availableRuns: uniqueRunOptions(dashboard.availableRuns),
  };
}

function isFocusAvailable(focus: ResearchFocus, selectedRunIds: string[], dashboard: DashboardPayload | null) {
  if (focus.kind === "strategy") {
    return selectedRunIds.includes(focus.runId);
  }

  if (focus.kind === "sector") {
    return selectedRunIds.some((runId) => {
      const timeSeriesMatch = (dashboard?.research.sectorWeightSeries[runId] ?? []).some(
        (series) => series.name === focus.sectorName,
      );
      const latestSnapshotMatch = (dashboard?.exposure.sectorWeights[runId] ?? []).some(
        (series) => series.name === focus.sectorName,
      );
      return timeSeriesMatch || latestSnapshotMatch;
    });
  }

  return true;
}

export function App() {
  const [runs, setRuns] = useState<RunOption[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [dashboard, setDashboard] = useState<DashboardPayload | null>(null);
  const [focus, setFocus] = useState<ResearchFocus>({ kind: "all-selected" });
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [dashboardError, setDashboardError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    setRunsLoading(true);

    void Promise.all([fetchRuns(), fetchSession().catch(() => ({ defaultSelectedRunIds: [] }))])
      .then(([nextRuns, bootstrap]) => {
        if (!isMounted) {
          return;
        }

        const normalizedRuns = uniqueRunOptions(nextRuns);
        setRuns(normalizedRuns);
        setRunsError(null);
        setSelectedRunIds(resolveInitialRunIds(normalizedRuns, bootstrap));
      })
      .catch((nextError: unknown) => {
        if (!isMounted) {
          return;
        }

        setRunsError(nextError instanceof Error ? nextError.message : "Failed to load saved runs.");
      })
      .finally(() => {
        if (!isMounted) {
          return;
        }

        setRunsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    if (selectedRunIds.length === 0) {
      setDashboard(null);
      setDashboardError(null);
      setFocus({ kind: "all-selected" });
      return () => {
        isMounted = false;
      };
    }

    const requestRunIds = orderSelectedRunIds(selectedRunIds, runs);
    setDashboard(null);
    setDashboardError(null);
    void fetchDashboard(requestRunIds)
      .then((nextDashboard) => {
        if (!isMounted) {
          return;
        }

        setDashboard(normalizeDashboardSelection(nextDashboard, runs));
        setDashboardError(null);
      })
      .catch((nextError: unknown) => {
        if (!isMounted) {
          return;
        }

        setDashboard(null);
        setDashboardError(nextError instanceof Error ? nextError.message : "Failed to load dashboard.");
      });

    return () => {
      isMounted = false;
    };
  }, [runs, selectedRunIds]);

  useEffect(() => {
    if (!isFocusAvailable(focus, selectedRunIds, dashboard)) {
      setFocus({ kind: "all-selected" });
    }
  }, [dashboard, focus, selectedRunIds]);

  return (
    <div className="dashboard-shell">
      <TopRail selectionCount={selectedRunIds.length} />
      <main className="dashboard-stage">
        {runsError ? <ErrorState message={runsError} /> : null}
        {dashboardError ? <ErrorState message={dashboardError} /> : null}
        {!runsLoading && !runsError && runs.length === 0 ? <EmptyState /> : null}
        {!runsLoading && runs.length > 0 ? (
          <RunSelector runs={runs} selectedRunIds={selectedRunIds} onToggle={toggleRun} />
        ) : null}
        {dashboard ? (
          <div className="cinema-workspace">
            <PerformanceStrip dashboard={dashboard} focus={focus} onFocusChange={setFocus} />
            <DiagnosticStrip dashboard={dashboard} focus={focus} onFocusChange={setFocus} />
            <ResearchWorkspace dashboard={dashboard} focus={focus} onFocusChange={setFocus} />
            <div className="detail-band">
              <ExposureBand dashboard={dashboard} focus={focus} onFocusChange={setFocus} />
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );

  function toggleRun(runId: string) {
    setSelectedRunIds((current) => {
      const nextSelection = current.includes(runId)
        ? current.filter((value) => value !== runId)
        : [...current, runId];

      return orderSelectedRunIds(nextSelection, runs);
    });
  }
}
