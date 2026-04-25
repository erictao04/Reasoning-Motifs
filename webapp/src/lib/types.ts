export type MotifDirection = "success" | "failure";
export type MotifScope = "question_local" | "corpus_global";

export interface StorySection {
  id: string;
  title: string;
  body: string;
}

export interface MotifRow {
  motif: string;
  tokens: string[];
  length: number;
  scope: MotifScope;
  direction: MotifDirection;
  success_count: number;
  failure_count: number;
  success_support: number;
  failure_support: number;
  support_difference: number;
  lift: number;
  log_odds_ratio: number;
  q_value?: number | null;
}

export interface TraceSummary {
  trace_id: string;
  question_id: string;
  sample_id: string;
  attempt_index: string;
  predicted_answer: string;
  is_correct: boolean;
  tokenized_trace: string;
  tokens: string[];
  token_count: number;
  reasoning_trace: string;
  matched_success_motifs: string[];
  matched_failure_motifs: string[];
}

export interface MotifBucket {
  available: boolean;
  reason?: string | null;
  success: MotifRow[];
  failure: MotifRow[];
}

export interface QuestionSummary {
  question_id: string;
  question_text: string;
  gold_answer: string;
  benchmark_name: string;
  total_traces: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
  avg_token_count: number;
  median_token_count: number;
  distinct_predicted_answers: number;
  local_motif_separation: number;
  top_success_motif?: string | null;
  top_failure_motif?: string | null;
  tags: string[];
}

export interface AnswerDistributionRow {
  answer: string;
  count: number;
  share: number;
}

export interface QuestionDetail extends QuestionSummary {
  pilot_question_uid: string;
  insights: string[];
  answer_distribution: AnswerDistributionRow[];
  local_motifs: MotifBucket;
  global_motifs: MotifBucket;
  representative_traces: Record<string, TraceSummary[]>;
  all_traces: TraceSummary[];
}

export interface CorpusOverview {
  corpus_label: string;
  num_questions: number;
  num_traces: number;
  num_success: number;
  num_failure: number;
  success_rate: number;
  avg_token_count: number;
  median_token_count: number;
  featured_question_ids: string[];
  story_sections: StorySection[];
  top_success_motifs: MotifRow[];
  top_failure_motifs: MotifRow[];
}
