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

export interface ResearchMotifCard {
  motif: string;
  motif_tokens: string[];
  motif_length: number;
  support: number;
  prevalence: number;
  question_coverage: number;
  success_present: number;
  failure_present: number;
  success_rate_present: number;
  success_rate_absent: number;
  support_difference: number;
  lift_overall_success: number;
  odds_ratio: number;
  odds_ratio_ci_low: number;
  odds_ratio_ci_high: number;
  fisher_p_value: number;
  fdr_q_value: number;
  avg_stage_position: number;
  stage_bucket: "early" | "middle" | "late";
  avg_trace_length_present: number;
  noise_cooccurrence_rate: number;
  representative_examples: Array<{
    question_id: string;
    sample_id: string;
    is_correct: boolean;
    predicted_answer: string;
    tokenized_trace: string;
  }>;
  top_neighbor_motifs: string[];
}

export interface MotifCardDatasetSummary {
  trace_count: number;
  question_count: number;
  question_trace_counts: Record<string, number>;
  success_count: number;
  failure_count: number;
  avg_token_count: number;
  median_token_count: number;
  max_token_count: number;
  rows_with_noise: number;
  rows_with_noise_rate: number;
  unique_token_count: number;
  token_type_counts: Record<string, number>;
  motif_definition: {
    type: string;
    min_len: number;
    max_len: number;
    excluded_substrings: string[];
  };
}

export interface QuestionMotifBundle {
  question_id: string;
  trace_count: number;
  success_count: number;
  failure_count: number;
  avg_token_count: number;
  rows_with_noise: number;
  local_top: ResearchMotifCard[];
  local_success_enriched: ResearchMotifCard[];
  local_failure_enriched: ResearchMotifCard[];
  visible_global_top: ResearchMotifCard[];
  visible_global_success: ResearchMotifCard[];
  visible_global_failure: ResearchMotifCard[];
}

export interface MotifCardFamilies {
  global_top: ResearchMotifCard[];
  success_enriched: ResearchMotifCard[];
  failure_enriched: ResearchMotifCard[];
}

export interface MotifCardExperiments {
  predictive: {
    motif_accuracy: number;
    motif_balanced_accuracy: number;
    motif_auc: number;
    motif_question_local_auc_mean: number;
    length_accuracy: number;
    length_balanced_accuracy: number;
    length_auc: number;
    length_question_local_auc_mean: number;
    avg_selected_feature_count: number;
  };
  coverage: {
    covered_trace_count: number;
    covered_trace_rate: number;
    covered_question_count: number;
    covered_question_rate: number;
  };
  compression: Array<{
    top_k: number;
    motif_auc: number;
    motif_question_local_auc: number;
    avg_feature_count: number;
  }>;
  length_matched: {
    max_length_gap: number;
    matched_pair_count: number;
    motif_pair_accuracy: number;
    length_pair_accuracy: number;
  };
}
