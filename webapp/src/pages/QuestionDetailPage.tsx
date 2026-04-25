import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { getQuestion } from "../lib/api";

export function QuestionDetailPage() {
  const { questionId = "" } = useParams();
  const questionQuery = useQuery({
    queryKey: ["question", questionId],
    queryFn: () => getQuestion(questionId),
    enabled: Boolean(questionId),
  });

  if (questionQuery.isLoading) {
    return <section className="panel">Loading question detail…</section>;
  }

  if (questionQuery.isError || !questionQuery.data) {
    return <section className="panel">Question detail could not be loaded.</section>;
  }

  return (
    <section className="stack-lg">
      <div className="panel">
        <p className="eyebrow">Question {questionQuery.data.question_id}</p>
        <h2>{questionQuery.data.question_text}</h2>
        <p>Gold answer: {questionQuery.data.gold_answer}</p>
      </div>
      <div className="metric-grid">
        <article className="metric-card">
          <span>Success</span>
          <strong>{questionQuery.data.success_count}</strong>
        </article>
        <article className="metric-card">
          <span>Failure</span>
          <strong>{questionQuery.data.failure_count}</strong>
        </article>
        <article className="metric-card">
          <span>Distinct answers</span>
          <strong>{questionQuery.data.distinct_predicted_answers}</strong>
        </article>
      </div>
      <div className="panel">
        <h3>What stands out</h3>
        <ul className="story-list">
          {questionQuery.data.insights.map((insight) => (
            <li key={insight}>{insight}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}
