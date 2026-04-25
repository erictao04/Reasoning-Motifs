import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getQuestions } from "../lib/api";

export function QuestionExplorerPage() {
  const questionsQuery = useQuery({
    queryKey: ["questions"],
    queryFn: getQuestions,
  });

  if (questionsQuery.isLoading) {
    return <section className="panel">Loading questions…</section>;
  }

  if (questionsQuery.isError || !questionsQuery.data) {
    return <section className="panel">Question explorer data is unavailable.</section>;
  }

  return (
    <section className="stack-lg">
      <div className="section-header">
        <div>
          <p className="eyebrow">Explorer</p>
          <h2>Question index</h2>
        </div>
      </div>
      <div className="card-grid">
        {questionsQuery.data.map((question) => (
          <article key={question.question_id} className="question-card panel">
            <div className="question-meta">
              <span>Q{question.question_id}</span>
              <span>{question.benchmark_name}</span>
            </div>
            <h3>{question.question_text}</h3>
            <p>
              {question.success_count} success / {question.failure_count} failure
            </p>
            <Link to={`/questions/${question.question_id}`} className="inline-link">
              Open question
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}
