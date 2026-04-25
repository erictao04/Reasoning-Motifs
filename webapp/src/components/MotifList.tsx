import type { MotifRow } from "../lib/types";

interface MotifListProps {
  title: string;
  motifs: MotifRow[];
  tone: "success" | "failure";
}

export function MotifList({ title, motifs, tone }: MotifListProps) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      {motifs.length === 0 ? (
        <p className="muted-copy">No motifs met the current evidence threshold.</p>
      ) : (
        <div className="motif-list">
          {motifs.map((motif) => (
            <article key={`${tone}-${motif.motif}`} className="motif-card">
              <div className="motif-sequence">
                {motif.tokens.map((token) => (
                  <span key={`${motif.motif}-${token}`} className={`motif-chip ${tone}`}>
                    {token}
                  </span>
                ))}
              </div>
              <dl className="motif-metrics">
                <div>
                  <dt>Success</dt>
                  <dd>{motif.success_count}</dd>
                </div>
                <div>
                  <dt>Failure</dt>
                  <dd>{motif.failure_count}</dd>
                </div>
                <div>
                  <dt>Lift</dt>
                  <dd>{motif.lift.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>Log odds</dt>
                  <dd>{motif.log_odds_ratio.toFixed(2)}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
