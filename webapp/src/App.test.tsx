import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

function renderApp(path: string) {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("App routes", () => {
  it("renders landing loading state", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    renderApp("/");
    expect(screen.getByText(/Loading the research story/i)).toBeInTheDocument();
  });

  it("renders question explorer loading state", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    renderApp("/questions");
    expect(screen.getByText(/Loading questions/i)).toBeInTheDocument();
  });

  it("renders question detail loading state", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    renderApp("/questions/11");
    expect(screen.getByText(/Loading question detail/i)).toBeInTheDocument();
  });
});
