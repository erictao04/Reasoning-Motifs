import { useQuery } from "@tanstack/react-query";
import { getOverview } from "../lib/api";

export function LandingPage() {
  const overviewQuery = useQuery({
    queryKey: ["overview"],
    queryFn: getOverview,
  });

  if (overviewQuery.isLoading) {
    return <section className="panel">Loading the research story…</section>;
  }

  if (overviewQuery.isError || !overviewQuery.data) {
    return <section className="panel">The overview payload could not be loaded.</section>;
  }

  return (
    <section className="stack-lg">
      <div className="hero">
        <p className="eyebrow">Method in Progress</p>
        <h2>Reasoning traces can be read as structural fingerprints.</h2>
        <p className="lede">
          This explorer turns sampled math reasoning traces into motifs, question-level
          comparisons, and a story about where the method could matter.
        </p>
      </div>
      <div className="metric-grid">
        <article className="metric-card">
          <span>Questions</span>
          <strong>{overviewQuery.data.num_questions}</strong>
        </article>
        <article className="metric-card">
          <span>Traces</span>
          <strong>{overviewQuery.data.num_traces}</strong>
        </article>
        <article className="metric-card">
          <span>Success Rate</span>
          <strong>{(overviewQuery.data.success_rate * 100).toFixed(1)}%</strong>
        </article>
      </div>
      <div className="panel">
        <h3>Story sections</h3>
        <ul className="story-list">
          {overviewQuery.data.story_sections.map((section) => (
            <li key={section.id}>
              <strong>{section.title}</strong>
              <p>{section.body}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
