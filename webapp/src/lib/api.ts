import type { CorpusOverview, QuestionDetail, QuestionSummary } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed for ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getOverview(): Promise<CorpusOverview> {
  return request<CorpusOverview>("/api/overview");
}

export function getQuestions(): Promise<QuestionSummary[]> {
  return request<QuestionSummary[]>("/api/questions");
}

export function getQuestion(questionId: string): Promise<QuestionDetail> {
  return request<QuestionDetail>(`/api/questions/${questionId}`);
}
