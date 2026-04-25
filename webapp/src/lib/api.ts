import type {
  CorpusOverview,
  MotifCardDatasetSummary,
  MotifCardExperiments,
  MotifCardFamilies,
  QuestionDetail,
  QuestionMotifBundle,
  QuestionSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const MOTIF_CARD_BASE =
  import.meta.env.VITE_MOTIF_CARD_BASE_PATH ?? "/motif-cards/gpt_oss_live_1_filtered";

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

async function requestStatic<T>(path: string): Promise<T> {
  const response = await fetch(`${MOTIF_CARD_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Static request failed for ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getMotifCardSummary(): Promise<MotifCardDatasetSummary> {
  return requestStatic<MotifCardDatasetSummary>("/dataset_summary.json");
}

export function getMotifCardFamilies(): Promise<MotifCardFamilies> {
  return requestStatic<MotifCardFamilies>("/motif_card_families.json");
}

export function getMotifCardExperiments(): Promise<MotifCardExperiments> {
  return requestStatic<MotifCardExperiments>("/motif_card_experiments.json");
}

export function getQuestionMotifBundles(): Promise<QuestionMotifBundle[]> {
  return requestStatic<QuestionMotifBundle[]>("/question_motif_cards.json");
}
