#!/usr/bin/env python3
"""Stage 6 / Experiment 5: per-question failure-mode clustering.

Inputs:
- Stage 2 per-question token deltas CSV (`per_question_token_deltas.csv`).

Outputs:
- cluster_signatures.csv
- question_clusters.csv
- umap.png
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import hdbscan
import umap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 6 failure-mode clustering from per-question token deltas."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Stage 2 per_question_token_deltas.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("research/stage6_exp5_failure_mode_clustering/results"),
        help="Output directory for Stage 6 artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed for UMAP and resampling.",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.10,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=8,
        help="HDBSCAN minimum cluster size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=4,
        help="HDBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10,
        help="Number of bootstrap resamples for cluster stability ARI.",
    )
    parser.add_argument(
        "--top-k-signature-tokens",
        type=int,
        default=5,
        help="How many top killer/lifesaver tokens to include per cluster signature.",
    )
    return parser.parse_args()


def read_per_question_deltas(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"question_id", "token", "delta_q"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {sorted(missing)}. "
                f"Found: {fieldnames}"
            )

        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        questions = set()
        tokens = set()
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            tok = (row.get("token") or "").strip()
            if not qid or not tok:
                continue
            try:
                dq = float(row.get("delta_q", ""))
            except ValueError:
                continue
            questions.add(qid)
            tokens.add(tok)
            values[(qid, tok)].append(dq)

    if not questions or not tokens:
        raise ValueError(f"No usable question/token rows found in: {path}")

    qids = sorted(questions, key=lambda q: (len(q), q))
    toks = sorted(tokens)
    q_index = {q: i for i, q in enumerate(qids)}
    t_index = {t: i for i, t in enumerate(toks)}
    matrix = np.zeros((len(qids), len(toks)), dtype=np.float64)

    for (qid, tok), deltas in values.items():
        matrix[q_index[qid], t_index[tok]] = float(sum(deltas) / len(deltas))
    return matrix, qids, toks


def fit_umap_hdbscan(
    features: np.ndarray,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=seed,
    )
    embedding = reducer.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding)
    return embedding, labels


def compute_cluster_signatures(
    matrix: np.ndarray,
    qids: list[str],
    tokens: list[str],
    labels: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows_signatures: list[dict[str, object]] = []
    rows_assignments: list[dict[str, object]] = []
    n_questions = len(qids)

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    for cluster_id in unique_labels:
        member_idx = np.where(labels == cluster_id)[0]
        if member_idx.size == 0:
            continue
        cluster_mat = matrix[member_idx, :]
        mean_deltas = cluster_mat.mean(axis=0)
        sorted_idx = np.argsort(mean_deltas)
        killer_idx = sorted_idx[:top_k]
        lifesaver_idx = sorted_idx[::-1][:top_k]

        killer_tokens = [tokens[i] for i in killer_idx]
        lifesaver_tokens = [tokens[i] for i in lifesaver_idx]
        killer_signature = "|".join(f"{tokens[i]}:{mean_deltas[i]:.4f}" for i in killer_idx)
        lifesaver_signature = "|".join(
            f"{tokens[i]}:{mean_deltas[i]:.4f}" for i in lifesaver_idx
        )
        sample_qids = [qids[i] for i in member_idx[:20]]
        label_name = (
            f"killer:{','.join(killer_tokens[:2])}"
            if cluster_id != -1
            else "noise_or_unclustered"
        )

        rows_signatures.append(
            {
                "cluster_id": cluster_id,
                "n_questions": int(member_idx.size),
                "question_fraction": float(member_idx.size / n_questions),
                "provisional_label": label_name,
                "dominant_killer_tokens": ",".join(killer_tokens),
                "dominant_killer_signature": killer_signature,
                "dominant_lifesaver_tokens": ",".join(lifesaver_tokens),
                "dominant_lifesaver_signature": lifesaver_signature,
                "example_question_ids": ",".join(sample_qids),
            }
        )

        for idx in member_idx:
            rows_assignments.append(
                {
                    "question_id": qids[idx],
                    "cluster_id": cluster_id,
                }
            )

    rows_signatures.sort(key=lambda r: (int(r["cluster_id"]) == -1, -int(r["n_questions"])))
    rows_assignments.sort(key=lambda r: (int(r["cluster_id"]), str(r["question_id"])))
    return rows_signatures, rows_assignments


def compute_bootstrap_stability(
    features: np.ndarray,
    base_labels: np.ndarray,
    seed: int,
    n_resamples: int,
    umap_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    aris: list[float] = []

    for i in range(n_resamples):
        sampled = rng.integers(0, n, size=n)
        unique_idx = np.unique(sampled)
        if unique_idx.size < 2:
            continue
        sub_features = features[unique_idx, :]
        _, sub_labels = fit_umap_hdbscan(
            sub_features,
            seed=seed + i + 1,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        base_sub = base_labels[unique_idx]
        ari = adjusted_rand_score(base_sub, sub_labels)
        aris.append(float(ari))

    if not aris:
        return {
            "n_resamples_requested": n_resamples,
            "n_resamples_valid": 0,
            "ari_mean": 0.0,
            "ari_median": 0.0,
            "ari_min": 0.0,
            "ari_max": 0.0,
            "ari_values": [],
        }

    return {
        "n_resamples_requested": n_resamples,
        "n_resamples_valid": len(aris),
        "ari_mean": float(np.mean(aris)),
        "ari_median": float(np.median(aris)),
        "ari_min": float(np.min(aris)),
        "ari_max": float(np.max(aris)),
        "ari_values": aris,
    }


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_umap(path: Path, embedding: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    cmap = plt.get_cmap("tab20")
    for i, cluster_id in enumerate(unique_labels):
        idx = np.where(labels == cluster_id)[0]
        color = "lightgray" if cluster_id == -1 else cmap(i % 20)
        marker = "x" if cluster_id == -1 else "o"
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=36,
            alpha=0.8,
            c=[color],
            marker=marker,
            label=f"cluster {cluster_id}",
        )

        if cluster_id != -1 and idx.size:
            cx = float(np.mean(embedding[idx, 0]))
            cy = float(np.mean(embedding[idx, 1]))
            ax.text(cx, cy, str(cluster_id), fontsize=10, fontweight="bold")

    ax.set_title("Stage 6 failure-mode clusters (UMAP + HDBSCAN)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matrix, qids, tokens = read_per_question_deltas(args.input)
    scaler = StandardScaler(with_mean=True, with_std=True)
    features = scaler.fit_transform(matrix)

    embedding, labels = fit_umap_hdbscan(
        features=features,
        seed=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    signatures, assignments = compute_cluster_signatures(
        matrix=matrix,
        qids=qids,
        tokens=tokens,
        labels=labels,
        top_k=args.top_k_signature_tokens,
    )
    stability = compute_bootstrap_stability(
        features=features,
        base_labels=labels,
        seed=args.seed,
        n_resamples=args.bootstrap_resamples,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        outdir / "cluster_signatures.csv",
        signatures,
        [
            "cluster_id",
            "n_questions",
            "question_fraction",
            "provisional_label",
            "dominant_killer_tokens",
            "dominant_killer_signature",
            "dominant_lifesaver_tokens",
            "dominant_lifesaver_signature",
            "example_question_ids",
        ],
    )
    write_csv(
        outdir / "question_clusters.csv",
        assignments,
        ["question_id", "cluster_id"],
    )
    plot_umap(outdir / "umap.png", embedding=embedding, labels=labels)

    cluster_sizes: dict[str, int] = {}
    for c in labels.tolist():
        key = str(int(c))
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1
    non_noise_clusters = sorted(c for c in cluster_sizes if c != "-1")

    summary = {
        "input_csv": str(args.input.resolve()),
        "n_questions": int(matrix.shape[0]),
        "n_tokens": int(matrix.shape[1]),
        "seed": args.seed,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "hdbscan_min_cluster_size": args.min_cluster_size,
        "hdbscan_min_samples": args.min_samples,
        "n_clusters_excluding_noise": len(non_noise_clusters),
        "cluster_sizes": cluster_sizes,
        "stability": stability,
        "stability_passes_ari_0_4": bool(stability["ari_mean"] >= 0.4),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote Stage 6 outputs to: {outdir.resolve()}")
    print(f"Questions clustered: {matrix.shape[0]}")
    print(f"Token features: {matrix.shape[1]}")
    print(f"Clusters (excluding noise): {len(non_noise_clusters)}")
    print(f"Bootstrap ARI mean ({stability['n_resamples_valid']} runs): {stability['ari_mean']:.4f}")


if __name__ == "__main__":
    main()
