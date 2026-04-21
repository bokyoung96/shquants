type ErrorStateProps = {
  message: string;
};

export function ErrorState({ message }: ErrorStateProps) {
  return (
    <section className="selector-panel error-state">
      <h2>Dashboard unavailable</h2>
      <p>{message}</p>
    </section>
  );
}
