import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { ResearchMotifCardList } from "../components/ResearchMotifCardList";
import { getQuestionMotifBundles } from "../lib/api";

export function MotifCardDetailPage() {
  const { questionId = "" } = useParams();
  const bundleQuery = useQuery({
    queryKey: ["question-motif-bundles"],
    queryFn: getQuestionMotifBundles,
  });

  if (bundleQuery.isLoading) {
    return <section className="panel">Loading question motifs…</section>;
  }

  if (bundleQuery.isError || !bundleQuery.data) {
    return <section className="panel">Question-local motif data could not be loaded.</section>;
  }

  const bundle = bundleQuery.data.find((entry) => entry.question_id === questionId);
  if (!bundle) {
    return <section className="panel">No local motif bundle was found for question {questionId}.</section>;
  }

  const successRate = bundle.success_count / Math.max(bundle.trace_count, 1);
  const noiseRate = bundle.rows_with_noise / Math.max(bundle.trace_count, 1);

  return (
    <section className="stack-xl">
      <section className="hero local-hero">
        <p className="eyebrow">Question {bundle.question_id}</p>
        <h2>Local motif bundle</h2>
        <p className="lede">
          This page only shows motifs mined from traces belonging to this one question.
          The goal is to understand the local structure of success and failure rather than
          comparing against corpus-wide signatures.
        </p>
        <Link to="/" className="inline-link back-link">
          Back to explorer
        </Link>
      </section>

      <div className="metric-grid metric-grid-compact">
        <article className="metric-card">
          <span>Traces</span>
          <strong>{bundle.trace_count}</strong>
        </article>
        <article className="metric-card">
          <span>Success</span>
          <strong>{bundle.success_count}</strong>
        </article>
        <article className="metric-card">
          <span>Failure</span>
          <strong>{bundle.failure_count}</strong>
        </article>
        <article className="metric-card">
          <span>Success rate</span>
          <strong>{(successRate * 100).toFixed(1)}%</strong>
        </article>
        <article className="metric-card">
          <span>Noise rate</span>
          <strong>{(noiseRate * 100).toFixed(1)}%</strong>
        </article>
        <article className="metric-card">
          <span>Avg token count</span>
          <strong>{bundle.avg_token_count.toFixed(1)}</strong>
        </article>
      </div>

      <section className="panel explainer-panel">
        <ul className="story-list">
          <li>
            <strong>Success motifs</strong> appear more often in successful traces for this
            question.
          </li>
          <li>
            <strong>Failure motifs</strong> appear more often in failed traces for this question,
            which makes them useful for diagnosing where reasoning tends to derail.
          </li>
        </ul>
      </section>

      <div className="two-column">
        <ResearchMotifCardList
          title="Success motifs"
          cards={bundle.local_success_enriched}
          tone="success"
          emptyCopy="No local success motifs met the current evidence threshold for this question."
        />
        <ResearchMotifCardList
          title="Failure motifs"
          cards={bundle.local_failure_enriched}
          tone="failure"
          emptyCopy="No local failure motifs met the current evidence threshold for this question."
        />
      </div>
    </section>
  );
}
