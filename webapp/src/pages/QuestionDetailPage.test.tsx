import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { QuestionDetailPage } from "./QuestionDetailPage";

const payload = {
  question_id: "68",
  question_text: "Complex interval question",
  gold_answer: "4",
  benchmark_name: "math_500",
  pilot_question_uid: "demo:q68",
  total_traces: 3,
  success_count: 2,
  failure_count: 1,
  success_rate: 0.67,
  avg_token_count: 6,
  median_token_count: 6,
  distinct_predicted_answers: 2,
  tags: ["mixed_outcomes"],
  insights: ["This question shows mixed outcomes."],
  answer_distribution: [
    { answer: "4", count: 2, share: 0.66 },
    { answer: "5", count: 1, share: 0.33 },
  ],
  local_motifs: {
    available: true,
    success: [
      {
        motif: "analyze|rewrite",
        tokens: ["analyze", "rewrite"],
        length: 2,
        scope: "question_local",
        direction: "success",
        success_count: 2,
        failure_count: 0,
        success_support: 1,
        failure_support: 0,
        support_difference: 1,
        lift: 2,
        log_odds_ratio: 1.4,
      },
    ],
    failure: [],
  },
  global_motifs: {
    available: false,
    reason: "No corpus-global motifs matched this question.",
    success: [],
    failure: [],
  },
  representative_traces: {
    success: [
      {
        trace_id: "68:1:1",
        question_id: "68",
        sample_id: "1",
        attempt_index: "1",
        predicted_answer: "4",
        is_correct: true,
        tokenized_trace: "analyze | rewrite | conclude",
        tokens: ["analyze", "rewrite", "conclude"],
        token_count: 3,
        reasoning_trace: "A successful trace.",
        matched_success_motifs: ["analyze|rewrite"],
        matched_failure_motifs: [],
      },
    ],
    failure: [
      {
        trace_id: "68:2:2",
        question_id: "68",
        sample_id: "2",
        attempt_index: "2",
        predicted_answer: "5",
        is_correct: false,
        tokenized_trace: "analyze | backtrack | conclude",
        tokens: ["analyze", "backtrack", "conclude"],
        token_count: 3,
        reasoning_trace: "A failed trace.",
        matched_success_motifs: [],
        matched_failure_motifs: ["analyze|backtrack"],
      },
    ],
  },
  all_traces: [
    {
      trace_id: "68:1:1",
      question_id: "68",
      sample_id: "1",
      attempt_index: "1",
      predicted_answer: "4",
      is_correct: true,
      tokenized_trace: "analyze | rewrite | conclude",
      tokens: ["analyze", "rewrite", "conclude"],
      token_count: 3,
      reasoning_trace: "A successful trace.",
      matched_success_motifs: ["analyze|rewrite"],
      matched_failure_motifs: [],
    },
    {
      trace_id: "68:2:2",
      question_id: "68",
      sample_id: "2",
      attempt_index: "2",
      predicted_answer: "5",
      is_correct: false,
      tokenized_trace: "analyze | backtrack | conclude",
      tokens: ["analyze", "backtrack", "conclude"],
      token_count: 3,
      reasoning_trace: "A failed trace.",
      matched_success_motifs: [],
      matched_failure_motifs: ["analyze|backtrack"],
    },
  ],
};

function renderPage() {
  const client = new QueryClient();
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    })),
  );

  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/questions/68"]}>
        <Routes>
          <Route path="/questions/:questionId" element={<QuestionDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("QuestionDetailPage", () => {
  it("renders motifs, insights, and representative traces", async () => {
    renderPage();
    expect(await screen.findByText(/Complex interval question/i)).toBeInTheDocument();
    expect(screen.getByText(/This question shows mixed outcomes/i)).toBeInTheDocument();
    expect(screen.getAllByText(/analyze\|rewrite/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/A successful trace/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/No corpus-global motifs matched this question/i)).toBeInTheDocument();
  });
});
