export function EmptyState() {
  return (
    <section className="selector-panel empty-state">
      <h2>No saved runs</h2>
      <p>
        Run <code>python run.py ...</code> first so the dashboard can load bundles from{" "}
        <code>results/backtests/</code>.
      </p>
    </section>
  );
}
