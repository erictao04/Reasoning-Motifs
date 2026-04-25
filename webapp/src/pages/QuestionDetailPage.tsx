import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getQuestion } from "../lib/api";
import { MotifList } from "../components/MotifList";
import { TraceCard } from "../components/TraceCard";

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

  const successTraces = questionQuery.data.all_traces.filter((trace) => trace.is_correct);
  const failureTraces = questionQuery.data.all_traces.filter((trace) => !trace.is_correct);
  const successAvgLength =
    successTraces.reduce((sum, trace) => sum + trace.token_count, 0) /
      Math.max(successTraces.length, 1);
  const failureAvgLength =
    failureTraces.reduce((sum, trace) => sum + trace.token_count, 0) /
      Math.max(failureTraces.length, 1);

  return (
    <section className="stack-lg">
      <section className="hero">
        <p className="eyebrow">Question {questionQuery.data.question_id}</p>
        <h2>{questionQuery.data.question_text}</h2>
        <p className="lede">
          Gold answer: <strong>{questionQuery.data.gold_answer}</strong>
        </p>
      </section>
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
        <article className="metric-card">
          <span>Avg success length</span>
          <strong>{successAvgLength.toFixed(1)}</strong>
        </article>
        <article className="metric-card">
          <span>Avg failure length</span>
          <strong>{failureAvgLength.toFixed(1)}</strong>
        </article>
      </div>
      <div className="two-column">
        <section className="panel">
          <h3>What stands out</h3>
          <ul className="story-list">
            {questionQuery.data.insights.map((insight) => (
              <li key={insight}>{insight}</li>
            ))}
          </ul>
        </section>
        <section className="panel">
          <h3>Answer distribution</h3>
          <div className="chart-frame">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={questionQuery.data.answer_distribution}>
                <CartesianGrid stroke="rgba(245,241,232,0.08)" vertical={false} />
                <XAxis dataKey="answer" stroke="rgba(245,241,232,0.6)" />
                <YAxis stroke="rgba(245,241,232,0.6)" />
                <Tooltip />
                <Bar dataKey="count" fill="#f3c969" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
      <div className="two-column">
        <section className="panel">
          <h3>Local motifs</h3>
          {questionQuery.data.local_motifs.available ? (
            <div className="stack-lg">
              <MotifList
                title="Success-enriched"
                motifs={questionQuery.data.local_motifs.success}
                tone="success"
              />
              <MotifList
                title="Failure-enriched"
                motifs={questionQuery.data.local_motifs.failure}
                tone="failure"
              />
            </div>
          ) : (
            <p className="muted-copy">{questionQuery.data.local_motifs.reason}</p>
          )}
        </section>
        <section className="panel">
          <h3>Corpus-global motifs present here</h3>
          {questionQuery.data.global_motifs.available ? (
            <div className="stack-lg">
              <MotifList
                title="Corpus success signatures"
                motifs={questionQuery.data.global_motifs.success}
                tone="success"
              />
              <MotifList
                title="Corpus failure signatures"
                motifs={questionQuery.data.global_motifs.failure}
                tone="failure"
              />
            </div>
          ) : (
            <p className="muted-copy">{questionQuery.data.global_motifs.reason}</p>
          )}
        </section>
      </div>
      <div className="panel">
        <h3>Representative traces</h3>
        <div className="trace-grid">
          {questionQuery.data.representative_traces.success.map((trace) => (
            <TraceCard key={trace.trace_id} trace={trace} tone="success" />
          ))}
          {questionQuery.data.representative_traces.failure.map((trace) => (
            <TraceCard key={trace.trace_id} trace={trace} tone="failure" />
          ))}
        </div>
      </div>
      <div className="panel">
        <h3>All traces</h3>
        <div className="trace-grid compact">
          {questionQuery.data.all_traces.map((trace) => (
            <TraceCard
              key={trace.trace_id}
              trace={trace}
              tone={trace.is_correct ? "success" : "failure"}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
