import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { QuestionExplorerPage } from "./QuestionExplorerPage";

const payload = [
  {
    question_id: "11",
    question_text: "Grid path question",
    gold_answer: "294",
    benchmark_name: "aime_2024",
    total_traces: 20,
    success_count: 19,
    failure_count: 1,
    success_rate: 0.95,
    avg_token_count: 10,
    median_token_count: 10,
    distinct_predicted_answers: 2,
    local_motif_separation: 0.12,
    top_success_motif: "count|case-split",
    top_failure_motif: null,
    tags: ["mixed_outcomes", "low_evidence"],
  },
  {
    question_id: "68",
    question_text: "Complex interval question",
    gold_answer: "4",
    benchmark_name: "math_500",
    total_traces: 24,
    success_count: 16,
    failure_count: 8,
    success_rate: 0.66,
    avg_token_count: 12,
    median_token_count: 12,
    distinct_predicted_answers: 4,
    local_motif_separation: 0.35,
    top_success_motif: "analyze|rewrite",
    top_failure_motif: "case-split|compute",
    tags: ["mixed_outcomes", "high_diversity"],
  },
  {
    question_id: "45",
    question_text: "Polynomial question",
    gold_answer: "2",
    benchmark_name: "math_500",
    total_traces: 10,
    success_count: 10,
    failure_count: 0,
    success_rate: 1,
    avg_token_count: 8,
    median_token_count: 8,
    distinct_predicted_answers: 1,
    local_motif_separation: 0,
    top_success_motif: null,
    top_failure_motif: null,
    tags: ["low_evidence"],
  },
];

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
      <MemoryRouter>
        <QuestionExplorerPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("QuestionExplorerPage", () => {
  it("filters by search and mixed toggle", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByText(/Grid path question/i)).toBeInTheDocument();
    expect(screen.getByText(/Complex interval question/i)).toBeInTheDocument();

    await user.type(screen.getByLabelText(/Search questions/i), "interval");
    expect(screen.queryByText(/Grid path question/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Complex interval question/i)).toBeInTheDocument();

    await user.clear(screen.getByLabelText(/Search questions/i));
    await user.click(screen.getByLabelText(/High answer diversity/i));
    expect(screen.queryByText(/Polynomial question/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Complex interval question/i)).toBeInTheDocument();
  });
});
