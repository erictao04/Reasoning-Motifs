from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
TRACES_ONLY_PATH = BASE_DIR / "pilot_traces_only.csv"
TRACES_WITH_CORRECTNESS_PATH = (
    BASE_DIR.parent / "question_independent_incorrect_see_correctness" / "pilot_traces.csv"
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "let",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "then",
    "therefore",
    "this",
    "to",
    "we",
    "when",
    "with",
    "x",
    "y",
    "z",
}


@dataclass
class SectionItem:
    trace_idx: int
    section_idx: int
    words: set[str]
    text: str
    cluster_id: int = -1


def load_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_sections(trace: str) -> List[str]:
    lines = trace.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: List[str] = []
    current: List[str] = []
    saw_step_marker = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                current.append("")
            continue

        lower = line.lower()
        is_step = (
            lower.startswith("step ")
            or lower.startswith("### step")
            or lower.startswith("**step")
        )

        if is_step and current:
            sections.append("\n".join(current).strip())
            current = [line]
            saw_step_marker = True
        else:
            current.append(line)
            if is_step:
                saw_step_marker = True

    if current:
        sections.append("\n".join(current).strip())

    if saw_step_marker and sections:
        return [s for s in sections if s]

    # Fallback: paragraph split when no explicit step markers are present.
    paragraphs: List[str] = []
    chunk: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if chunk:
                paragraphs.append(" ".join(chunk).strip())
                chunk = []
        else:
            chunk.append(line)
    if chunk:
        paragraphs.append(" ".join(chunk).strip())
    return [p for p in paragraphs if p]


def normalize_words(text: str) -> List[str]:
    cleaned_chars: List[str] = []
    for ch in text.lower():
        cleaned_chars.append(ch if ch.isalnum() else " ")
    words = "".join(cleaned_chars).split()
    filtered: List[str] = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) <= 2:
            continue
        filtered.append(word)
    return filtered


def top_word_set(text: str, top_k: int = 6) -> set[str]:
    counts = Counter(normalize_words(text))
    if not counts:
        return {"misc"}
    return {word for word, _ in counts.most_common(top_k)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def induce_clusters(items: List[SectionItem], threshold: float = 0.5) -> List[dict]:
    clusters: List[dict] = []
    for item in items:
        best_idx = -1
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            score = jaccard(item.words, cluster["signature"])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            cluster = clusters[best_idx]
            cluster["items"].append(item)
            cluster["counter"].update(item.words)
            cluster["signature"] = {w for w, _ in cluster["counter"].most_common(8)}
            item.cluster_id = best_idx
        else:
            counter = Counter(item.words)
            new_cluster = {
                "items": [item],
                "counter": counter,
                "signature": {w for w, _ in counter.most_common(8)},
            }
            clusters.append(new_cluster)
            item.cluster_id = len(clusters) - 1
    return clusters


def token_name(question_id: str, token_idx: int, cluster_counter: Counter) -> str:
    keywords = [word for word, _ in cluster_counter.most_common(3)]
    if not keywords:
        keywords = ["misc"]
    suffix = "_".join(keywords)
    return f"Q{question_id}_T{token_idx:02d}_{suffix}"


def build_correctness_map(rows: Iterable[dict]) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    for row in rows:
        key = (str(row.get("question_id", "")), str(row.get("sample_id", "")), str(row.get("attempt_index", "")))
        out[key] = str(row.get("is_correct", ""))
    return out


def sort_key(row: dict) -> Tuple[Tuple[int, object], Tuple[int, object]]:
    sample_id = str(row.get("sample_id", ""))
    attempt_index = str(row.get("attempt_index", ""))
    return as_int_sort_key(sample_id), as_int_sort_key(attempt_index)


def as_int_sort_key(value: str) -> Tuple[int, object]:
    try:
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, value


def process_question(
    question_id: str,
    question_rows: List[dict],
    correctness_map: Dict[Tuple[str, str, str], str],
) -> Tuple[int, int]:
    traces = [str(r.get("reasoning_trace", "")) for r in question_rows]
    sections_by_trace = [split_sections(t) for t in traces]

    items: List[SectionItem] = []
    for t_idx, sections in enumerate(sections_by_trace):
        for s_idx, section in enumerate(sections):
            items.append(
                SectionItem(
                    trace_idx=t_idx,
                    section_idx=s_idx,
                    words=top_word_set(section),
                    text=section,
                )
            )

    clusters = induce_clusters(items, threshold=0.5)
    cluster_order = sorted(
        range(len(clusters)),
        key=lambda idx: len(clusters[idx]["items"]),
        reverse=True,
    )

    cluster_to_token: Dict[int, str] = {}
    for i, cluster_id in enumerate(cluster_order, start=1):
        cluster_to_token[cluster_id] = token_name(question_id, i, clusters[cluster_id]["counter"])

    trace_token_map: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for item in items:
        trace_token_map[item.trace_idx].append((item.section_idx, cluster_to_token[item.cluster_id]))

    tokenized_rows: List[dict] = []
    missing_correctness = 0
    for t_idx, row in enumerate(question_rows):
        ordered_tokens = [token for _, token in sorted(trace_token_map[t_idx], key=lambda x: x[0])]
        key = (str(row.get("question_id", "")), str(row.get("sample_id", "")), str(row.get("attempt_index", "")))
        is_correct = correctness_map.get(key, "")
        if is_correct == "":
            missing_correctness += 1
        tokenized_rows.append(
            {
                "question_id": row.get("question_id", ""),
                "sample_id": row.get("sample_id", ""),
                "attempt_index": row.get("attempt_index", ""),
                "tokenized_trace": " > ".join(ordered_tokens),
                "is_correct": is_correct,
            }
        )

    tokenized_rows.sort(key=sort_key)
    tokenized_path = BASE_DIR / f"question_{question_id}_tokenized_model.csv"
    with tokenized_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "sample_id", "attempt_index", "tokenized_trace", "is_correct"],
        )
        writer.writeheader()
        writer.writerows(tokenized_rows)

    dictionary_rows: List[dict] = []
    for cluster_id in cluster_order:
        token = cluster_to_token[cluster_id]
        counter = clusters[cluster_id]["counter"]
        keywords = [word for word, _ in counter.most_common(8)]
        dictionary_rows.append(
            {
                "token": token,
                "num_sections": len(clusters[cluster_id]["items"]),
                "keywords": ", ".join(keywords),
            }
        )

    dict_path = BASE_DIR / f"question_{question_id}_token_dictionary.csv"
    with dict_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "num_sections", "keywords"])
        writer.writeheader()
        writer.writerows(dictionary_rows)

    return len(tokenized_rows), missing_correctness


def main() -> None:
    traces_only_rows = load_csv(TRACES_ONLY_PATH)
    traces_with_correctness_rows = load_csv(TRACES_WITH_CORRECTNESS_PATH)
    correctness_map = build_correctness_map(traces_with_correctness_rows)

    by_question: Dict[str, List[dict]] = defaultdict(list)
    for row in traces_only_rows:
        by_question[str(row.get("question_id", ""))].append(row)

    total_rows = 0
    total_missing_correctness = 0
    question_ids = sorted(by_question.keys(), key=as_int_sort_key)
    for qid in question_ids:
        rows = by_question[qid]
        count, missing = process_question(qid, rows, correctness_map)
        total_rows += count
        total_missing_correctness += missing
        print(f"question_id={qid}: wrote {count} tokenized rows, missing is_correct={missing}")

    print(
        f"Done. questions={len(question_ids)}, total_rows={total_rows}, "
        f"total_missing_is_correct={total_missing_correctness}"
    )


if __name__ == "__main__":
    main()
