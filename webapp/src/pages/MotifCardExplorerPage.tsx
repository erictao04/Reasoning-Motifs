import { useDeferredValue, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getMotifCardSummary, getQuestionMotifBundles } from "../lib/api";

function sortBundles(
  bundles: Awaited<ReturnType<typeof getQuestionMotifBundles>>,
  sortKey: string,
) {
  return [...bundles].sort((left, right) => {
    const leftNoise = left.rows_with_noise / Math.max(left.trace_count, 1);
    const rightNoise = right.rows_with_noise / Math.max(right.trace_count, 1);
    const leftSuccessRate = left.success_count / Math.max(left.trace_count, 1);
    const rightSuccessRate = right.success_count / Math.max(right.trace_count, 1);
    const leftMixedness = 1 - Math.abs((leftSuccessRate * 2) - 1);
    const rightMixedness = 1 - Math.abs((rightSuccessRate * 2) - 1);
    if (sortKey === "noise") {
      return rightNoise - leftNoise;
    }
    if (sortKey === "success") {
      return leftSuccessRate - rightSuccessRate;
    }
    if (sortKey === "mixed") {
      if (leftMixedness !== rightMixedness) {
        return rightMixedness - leftMixedness;
      }
      return right.failure_count - left.failure_count;
    }
    return right.failure_count - left.failure_count;
  });
}

export function MotifCardExplorerPage() {
  const summaryQuery = useQuery({
    queryKey: ["motif-card-summary"],
    queryFn: getMotifCardSummary,
  });
  const bundlesQuery = useQuery({
    queryKey: ["question-motif-bundles"],
    queryFn: getQuestionMotifBundles,
  });
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("mixed");
  const deferredSearch = useDeferredValue(search);

  if (summaryQuery.isLoading || bundlesQuery.isLoading) {
    return <section className="panel">Loading local motif explorer…</section>;
  }

  if (summaryQuery.isError || bundlesQuery.isError || !summaryQuery.data || !bundlesQuery.data) {
    return <section className="panel">Local motif explorer data is unavailable.</section>;
  }

  const normalizedSearch = deferredSearch.trim().toLowerCase();
  const filteredBundles = sortBundles(
    bundlesQuery.data.filter((bundle) => {
      if (normalizedSearch.length === 0) {
        return true;
      }
      return (
        bundle.question_id.includes(normalizedSearch) ||
        bundle.local_top.some((card) => card.motif.toLowerCase().includes(normalizedSearch)) ||
        bundle.local_success_enriched.some((card) =>
          card.motif.toLowerCase().includes(normalizedSearch),
        ) ||
        bundle.local_failure_enriched.some((card) =>
          card.motif.toLowerCase().includes(normalizedSearch),
        )
      );
    }),
    sortKey,
  );

  const summary = summaryQuery.data;

  return (
    <section className="stack-xl">
      <section className="hero local-hero">
        <p className="eyebrow">Project Focus</p>
        <h2>Question-local motifs as a lens on reasoning behavior.</h2>
        <p className="lede">
          Each question gets its own compact motif bundle: the most common local
          subsequences, plus success- and failure-enriched patterns that only make sense
          inside that question’s trace distribution.
        </p>
      </section>

      <div className="metric-grid metric-grid-compact">
        <article className="metric-card">
          <span>Questions</span>
          <strong>{summary.question_count}</strong>
        </article>
        <article className="metric-card">
          <span>Traces</span>
          <strong>{summary.trace_count}</strong>
        </article>
        <article className="metric-card">
          <span>Success</span>
          <strong>{summary.success_count}</strong>
        </article>
        <article className="metric-card">
          <span>Failure</span>
          <strong>{summary.failure_count}</strong>
        </article>
        <article className="metric-card">
          <span>Noise rows</span>
          <strong>{(summary.rows_with_noise_rate * 100).toFixed(1)}%</strong>
        </article>
      </div>

      <section className="panel explainer-panel">
        <div className="use-grid">
          <article className="use-card">
            <h4>Why local?</h4>
            <p>
              A motif can mean different things across questions. Local bundles keep the
              comparison anchored to one problem’s own success and failure traces.
            </p>
          </article>
          <article className="use-card">
            <h4>What is a motif?</h4>
            <p>
              Here it is a contiguous subsequence of token length 1 to 3, mined from the
              tokenized traces for that question.
            </p>
          </article>
          <article className="use-card">
            <h4>How to use this view</h4>
            <p>
              Open a question to inspect its local top motifs, then compare which patterns
              lean toward successful or failed traces.
            </p>
          </article>
        </div>
      </section>

      <section className="panel explorer-controls local-controls">
        <label className="control-group">
          <span>Search questions or motifs</span>
          <input
            aria-label="Search local motif bundles"
            placeholder="Question id or motif phrase"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
        <label className="control-group">
          <span>Sort</span>
          <select
            aria-label="Sort local motif bundles"
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value)}
          >
            <option value="mixed">Most mixed</option>
            <option value="failures">Most failure traces</option>
            <option value="noise">Highest noise rate</option>
            <option value="success">Lowest success rate</option>
          </select>
        </label>
      </section>

      <p className="results-copy">
        Showing {filteredBundles.length} of {bundlesQuery.data.length} questions.
      </p>

      <div className="local-question-list">
        {filteredBundles.map((bundle) => {
          const noiseRate = bundle.rows_with_noise / Math.max(bundle.trace_count, 1);
          const successRate = bundle.success_count / Math.max(bundle.trace_count, 1);
          return (
            <article key={bundle.question_id} className="local-question-card">
              <div className="local-question-head">
                <div>
                  <p className="eyebrow">Question {bundle.question_id}</p>
                  <h3>Local motif bundle</h3>
                </div>
                <Link to={`/questions/${bundle.question_id}`} className="cta-link subtle">
                  Open
                </Link>
              </div>

              <dl className="question-stats">
                <div>
                  <dt>Success</dt>
                  <dd>{bundle.success_count}</dd>
                </div>
                <div>
                  <dt>Failure</dt>
                  <dd>{bundle.failure_count}</dd>
                </div>
                <div>
                  <dt>Success rate</dt>
                  <dd>{(successRate * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Noise rate</dt>
                  <dd>{(noiseRate * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt>Avg token count</dt>
                  <dd>{bundle.avg_token_count.toFixed(1)}</dd>
                </div>
              </dl>

              <div className="local-preview-grid">
                <div>
                  <p className="preview-label">Success motifs</p>
                  <div className="trace-match-group">
                    {bundle.local_success_enriched.slice(0, 4).map((card) => (
                      <span
                        key={`${bundle.question_id}-success-${card.motif}`}
                        className="motif-hit"
                      >
                        {card.motif}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="preview-label">Failure motifs</p>
                  <div className="trace-match-group">
                    {bundle.local_failure_enriched.slice(0, 3).map((card) => (
                      <span
                        key={`${bundle.question_id}-fail-${card.motif}`}
                        className="motif-hit failure"
                      >
                        {card.motif}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
