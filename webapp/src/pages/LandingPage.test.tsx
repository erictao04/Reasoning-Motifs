import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { LandingPage } from "./LandingPage";

const payload = {
  corpus_label: "Curated pilot reasoning traces",
  num_questions: 22,
  num_traces: 251,
  num_success: 180,
  num_failure: 71,
  success_rate: 0.71,
  avg_token_count: 12,
  median_token_count: 11,
  featured_question_ids: ["11", "68"],
  story_sections: [
    { id: "hypothesis", title: "Reasoning fingerprints", body: "Motifs expose structural differences." },
    { id: "method", title: "Method shape", body: "Join traces, mine motifs, inspect examples." },
  ],
  top_success_motifs: [
    {
      motif: "analyze|instantiate",
      tokens: ["analyze", "instantiate"],
      length: 2,
      scope: "corpus_global",
      direction: "success",
      success_count: 12,
      failure_count: 1,
      success_support: 0.2,
      failure_support: 0.01,
      support_difference: 0.19,
      lift: 2.1,
      log_odds_ratio: 1.5,
    },
  ],
  top_failure_motifs: [
    {
      motif: "case-split|compute",
      tokens: ["case-split", "compute"],
      length: 2,
      scope: "corpus_global",
      direction: "failure",
      success_count: 1,
      failure_count: 10,
      success_support: 0.01,
      failure_support: 0.2,
      support_difference: -0.19,
      lift: 0.1,
      log_odds_ratio: -1.7,
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
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LandingPage", () => {
  it("renders the story, uses, and featured case studies", async () => {
    renderPage();
    expect(await screen.findByText(/Reasoning traces can be read as structural fingerprints/i)).toBeInTheDocument();
    expect(screen.getByText(/Model debugging/i)).toBeInTheDocument();
    expect(screen.getByText(/Question 11/i)).toBeInTheDocument();
    expect(screen.getByText(/Top corpus success motifs/i)).toBeInTheDocument();
  });
});
