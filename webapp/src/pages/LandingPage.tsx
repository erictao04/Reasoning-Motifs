import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getOverview } from "../lib/api";
import { MotifList } from "../components/MotifList";

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
      <div className="story-grid">
        {overviewQuery.data.story_sections.map((section) => (
          <article key={section.id} className="panel story-card">
            <p className="eyebrow">{section.title}</p>
            <p>{section.body}</p>
          </article>
        ))}
      </div>
      <section className="panel">
        <h3>Why this could matter</h3>
        <div className="use-grid">
          <article className="use-card">
            <h4>Model debugging</h4>
            <p>Motifs can expose recurring failure paths that raw accuracy hides.</p>
          </article>
          <article className="use-card">
            <h4>Benchmark diagnosis</h4>
            <p>Questions with mixed outcomes become visible as distinct reasoning regimes.</p>
          </article>
          <article className="use-card">
            <h4>Intervention targeting</h4>
            <p>Stable failure motifs suggest where prompting or training changes might bite.</p>
          </article>
          <article className="use-card">
            <h4>Interpretability</h4>
            <p>Reasoning structure becomes more legible than final answers alone.</p>
          </article>
        </div>
      </section>
      <section className="panel">
        <div className="section-split">
          <div>
            <p className="eyebrow">New artifact</p>
            <h3>Motif-card explorer</h3>
            <p className="lede">
              Browse question-level motif bundles built from the completed GPT-OSS run,
              with filtered global cards, local success and failure signatures, and
              compact evidence for why the abstraction helps.
            </p>
          </div>
          <Link to="/motif-cards" className="cta-link">
            Open motif cards
          </Link>
        </div>
      </section>
      <div className="two-column">
        <MotifList
          title="Top corpus success motifs"
          motifs={overviewQuery.data.top_success_motifs.slice(0, 6)}
          tone="success"
        />
        <MotifList
          title="Top corpus failure motifs"
          motifs={overviewQuery.data.top_failure_motifs.slice(0, 6)}
          tone="failure"
        />
      </div>
      <section className="panel">
        <h3>Featured case studies</h3>
        <div className="card-grid">
          {overviewQuery.data.featured_question_ids.map((questionId) => (
            <article key={questionId} className="question-card panel">
              <p className="eyebrow">Case study</p>
              <h3>Question {questionId}</h3>
              <p>
                Jump directly into a question page where motif evidence and raw traces
                can be inspected together.
              </p>
              <Link to={`/questions/${questionId}`} className="inline-link">
                Open question
              </Link>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}
