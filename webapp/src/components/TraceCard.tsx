import type { TraceSummary } from "../lib/types";

interface TraceCardProps {
  trace: TraceSummary;
  tone: "success" | "failure";
}

export function TraceCard({ trace, tone }: TraceCardProps) {
  const successTokens = new Set(
    trace.matched_success_motifs.flatMap((motif) => motif.split("|")),
  );
  const failureTokens = new Set(
    trace.matched_failure_motifs.flatMap((motif) => motif.split("|")),
  );

  return (
    <article className={`panel trace-card ${tone}`}>
      <div className="trace-card-header">
        <div>
          <p className="eyebrow">
            sample {trace.sample_id} · attempt {trace.attempt_index}
          </p>
          <h4>{trace.predicted_answer}</h4>
        </div>
        <span className={`tag ${tone}`}>{tone}</span>
      </div>
      <div className="trace-token-row">
        {trace.tokens.map((token, index) => {
          let flavor = "";
          if (successTokens.has(token)) {
            flavor = "success";
          } else if (failureTokens.has(token)) {
            flavor = "failure";
          }
          return (
            <span key={`${trace.trace_id}-${token}-${index}`} className={`trace-token ${flavor}`}>
              {token}
            </span>
          );
        })}
      </div>
      <div className="trace-match-group">
        {trace.matched_success_motifs.map((motif) => (
          <span key={`${trace.trace_id}-s-${motif}`} className="motif-hit success">
            {motif}
          </span>
        ))}
        {trace.matched_failure_motifs.map((motif) => (
          <span key={`${trace.trace_id}-f-${motif}`} className="motif-hit failure">
            {motif}
          </span>
        ))}
      </div>
      <details className="trace-details">
        <summary>Show raw reasoning trace</summary>
        <pre>{trace.reasoning_trace}</pre>
      </details>
    </article>
  );
}
