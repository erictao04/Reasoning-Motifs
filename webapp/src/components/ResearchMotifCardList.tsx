import type { ResearchMotifCard } from "../lib/types";

interface ResearchMotifCardListProps {
  title: string;
  cards: ResearchMotifCard[];
  tone: "success" | "failure" | "neutral";
  emptyCopy?: string;
}

function toneClass(tone: "success" | "failure" | "neutral") {
  if (tone === "neutral") {
    return "";
  }
  return tone;
}

export function ResearchMotifCardList({
  title,
  cards,
  tone,
  emptyCopy = "No motif cards met the current evidence threshold.",
}: ResearchMotifCardListProps) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      {cards.length === 0 ? (
        <p className="muted-copy">{emptyCopy}</p>
      ) : (
        <div className="motif-list">
          {cards.map((card) => (
            <article key={`${title}-${card.motif}`} className="motif-card research-card">
              <div className="motif-card-topline">
                <div className="motif-sequence">
                  {card.motif_tokens.map((token) => (
                    <span
                      key={`${card.motif}-${token}`}
                      className={`motif-chip ${toneClass(tone)}`}
                    >
                      {token}
                    </span>
                  ))}
                </div>
                <span className={`stage-pill ${card.stage_bucket}`}>{card.stage_bucket}</span>
              </div>
              <dl className="motif-metrics motif-metrics-wide">
                <div>
                  <dt>Support</dt>
                  <dd>{card.support}</dd>
                </div>
                <div>
                  <dt>Questions</dt>
                  <dd>{card.question_coverage}</dd>
                </div>
                <div>
                  <dt>Success if present</dt>
                  <dd>{(card.success_rate_present * 100).toFixed(0)}%</dd>
                </div>
                <div>
                  <dt>Success if absent</dt>
                  <dd>{(card.success_rate_absent * 100).toFixed(0)}%</dd>
                </div>
                <div>
                  <dt>Lift</dt>
                  <dd>{card.lift_overall_success.toFixed(2)}</dd>
                </div>
                <div>
                  <dt>Noise</dt>
                  <dd>{(card.noise_cooccurrence_rate * 100).toFixed(0)}%</dd>
                </div>
              </dl>
              {card.representative_examples.length > 0 ? (
                <details className="trace-details">
                  <summary>Representative examples</summary>
                  <div className="example-stack">
                    {card.representative_examples.map((example) => (
                      <article
                        key={`${card.motif}-${example.question_id}-${example.sample_id}`}
                        className="example-card"
                      >
                        <p className="eyebrow">
                          Q{example.question_id} · sample {example.sample_id}
                        </p>
                        <p className="example-answer">
                          {example.is_correct ? "Correct" : "Incorrect"} · {example.predicted_answer}
                        </p>
                        <div className="trace-token-row compact">
                          {example.tokenized_trace.split(" ").map((token, index) => (
                            <span
                              key={`${card.motif}-${example.sample_id}-${token}-${index}`}
                              className="trace-token"
                            >
                              {token}
                            </span>
                          ))}
                        </div>
                      </article>
                    ))}
                  </div>
                </details>
              ) : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
