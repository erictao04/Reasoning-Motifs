import { useDeferredValue, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getQuestions } from "../lib/api";
import type { QuestionSummary } from "../lib/types";

function sortQuestions(questions: QuestionSummary[], sortKey: string) {
  return [...questions].sort((left, right) => {
    if (sortKey === "success_rate") {
      return right.success_rate - left.success_rate;
    }
    if (sortKey === "trace_count") {
      return right.total_traces - left.total_traces;
    }
    const leftMixed = left.tags.includes("mixed_outcomes") ? 1 : 0;
    const rightMixed = right.tags.includes("mixed_outcomes") ? 1 : 0;
    if (leftMixed !== rightMixed) {
      return rightMixed - leftMixed;
    }
    return right.local_motif_separation - left.local_motif_separation;
  });
}

export function QuestionExplorerPage() {
  const questionsQuery = useQuery({
    queryKey: ["questions"],
    queryFn: getQuestions,
  });
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("signal");
  const [mixedOnly, setMixedOnly] = useState(false);
  const [highDiversityOnly, setHighDiversityOnly] = useState(false);
  const deferredSearch = useDeferredValue(search);

  if (questionsQuery.isLoading) {
    return <section className="panel">Loading questions…</section>;
  }

  if (questionsQuery.isError || !questionsQuery.data) {
    return <section className="panel">Question explorer data is unavailable.</section>;
  }

  const normalizedSearch = deferredSearch.trim().toLowerCase();
  const filteredQuestions = sortQuestions(
    questionsQuery.data.filter((question) => {
      const matchesSearch =
        normalizedSearch.length === 0 ||
        question.question_text.toLowerCase().includes(normalizedSearch) ||
        question.question_id.includes(normalizedSearch);
      const matchesMixed = !mixedOnly || question.tags.includes("mixed_outcomes");
      const matchesDiversity =
        !highDiversityOnly || question.tags.includes("high_diversity");
      return matchesSearch && matchesMixed && matchesDiversity;
    }),
    sortKey,
  );

  return (
    <section className="stack-lg">
      <div className="section-header">
        <div>
          <p className="eyebrow">Explorer</p>
          <h2>Question index</h2>
        </div>
      </div>
      <section className="panel explorer-controls">
        <label className="control-group">
          <span>Search</span>
          <input
            aria-label="Search questions"
            placeholder="Question id or phrase"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>
        <label className="control-group">
          <span>Sort</span>
          <select
            aria-label="Sort questions"
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value)}
          >
            <option value="signal">Mixed first, then motif separation</option>
            <option value="success_rate">Highest success rate</option>
            <option value="trace_count">Most traces</option>
          </select>
        </label>
        <label className="toggle-pill">
          <input
            type="checkbox"
            checked={mixedOnly}
            onChange={(event) => setMixedOnly(event.target.checked)}
          />
          Mixed outcomes only
        </label>
        <label className="toggle-pill">
          <input
            type="checkbox"
            checked={highDiversityOnly}
            onChange={(event) => setHighDiversityOnly(event.target.checked)}
          />
          High answer diversity
        </label>
      </section>
      <p className="results-copy">
        Showing {filteredQuestions.length} of {questionsQuery.data.length} questions.
      </p>
      <div className="card-grid">
        {filteredQuestions.map((question) => (
          <article
            key={question.question_id}
            className={`question-card panel ${question.tags.includes("low_evidence") ? "is-low-evidence" : ""}`}
          >
            <div className="question-meta">
              <span>Q{question.question_id}</span>
              <span>{question.benchmark_name}</span>
            </div>
            <h3>{question.question_text}</h3>
            <div className="tag-row">
              {question.tags.map((tag) => (
                <span key={tag} className={`tag ${tag}`}>
                  {tag.replace(/_/g, " ")}
                </span>
              ))}
            </div>
            <dl className="question-stats">
              <div>
                <dt>Outcome split</dt>
                <dd>
                  {question.success_count} success / {question.failure_count} failure
                </dd>
              </div>
              <div>
                <dt>Success rate</dt>
                <dd>{(question.success_rate * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Motif separation</dt>
                <dd>{question.local_motif_separation.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Distinct answers</dt>
                <dd>{question.distinct_predicted_answers}</dd>
              </div>
            </dl>
            <Link to={`/questions/${question.question_id}`} className="inline-link">
              Open question
            </Link>
          </article>
        ))}
      </div>
      {filteredQuestions.length === 0 ? (
        <section className="panel">No questions matched the current explorer filters.</section>
      ) : null}
    </section>
  );
}
