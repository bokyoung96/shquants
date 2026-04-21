type TopRailProps = {
  selectionCount: number;
};

export function TopRail({ selectionCount }: TopRailProps) {
  return (
    <header className="top-rail">
      <div className="brand-lockup">
        <span className="brand-mark">1W1A</span>
        <span className="brand-subtitle">Live Performance</span>
      </div>
      <div className="top-rail-meta">
        <span className="top-rail-pill">{selectionCount} selected</span>
        <span className="top-rail-pill">Comparison workspace</span>
      </div>
    </header>
  );
}
