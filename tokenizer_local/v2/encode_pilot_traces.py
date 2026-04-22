from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR.parent / "v1" / "pilot_traces.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "tokenized_pilot_traces.csv"


# Canonical reasoning actions (same family as v1 for compatibility).
CANONICAL_REASONING_TOKENS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "backtrack",
            (
                r"\bwait\b",
                r"\bon second thought\b",
                r"\binstead\b",
                r"\bno,\b",
                r"\bthat (?:formula|approach|claim) (?:is|was) (?:not|wrong)\b",
                r"\blet'?s verify\b",
                r"\bre-verify\b",
            ),
        ),
        (
            "case-split",
            (
                r"\bcase\b",
                r"\bcases\b",
                r"\bthere are two cases\b",
                r"\bmutually exclusive\b",
                r"\bwithout loss of generality\b",
            ),
        ),
        (
            "check-constraint",
            (
                r"\bcheck\b",
                r"\bverify\b",
                r"\bdouble check\b",
                r"\bcondition(?:s)?\b",
                r"\bconstraint(?:s)?\b",
                r"\bsatisfied\b",
                r"\bholds\b",
                r"\bvalid\b",
                r"\bprime\?\b",
            ),
        ),
        (
            "instantiate",
            (
                r"\blet\b",
                r"\bdefine\b",
                r"\bdenote\b",
                r"\bassume\b",
                r"\bset\b",
                r"\bparameterize\b",
                r"\bwrite\s+\w+\s*=",
            ),
        ),
        (
            "analyze",
            (
                r"\bidentify\b",
                r"\banalyze\b",
                r"\bobserve\b",
                r"\bnotice\b",
                r"\brecognize\b",
                r"\bthe given\b",
                r"\bwe need to find\b",
                r"\bthe problem states\b",
                r"\bdetermine\b",
            ),
        ),
        (
            "apply-formula",
            (
                r"\bformula\b",
                r"\bidentity\b",
                r"\bheron'?s\b",
                r"\beuler'?s\b",
                r"\buse the\b",
                r"\bis given by\b",
                r"\bthe area of\b",
                r"\bthe volume of\b",
                r"\bthe probability of\b",
            ),
        ),
        (
            "rewrite",
            (
                r"\brewrite\b",
                r"\brearrange\b",
                r"\bexpress\b",
                r"\bfactor(?:ize)?\b",
                r"\bexpand\b",
                r"\bsimplify\b",
                r"\brationalize\b",
                r"\bcomplete the square\b",
                r"\bgroup the terms\b",
                r"\bseparate\b",
            ),
        ),
        (
            "substitute",
            (
                r"\bsubstitute\b",
                r"\bplug(?:ging)?(?:\s+in)?\b",
                r"\bsubstitute back\b",
                r"\bnow plug\b",
                r"\bnow substitute\b",
            ),
        ),
        (
            "count",
            (
                r"\bcount\b",
                r"\bnumber of\b",
                r"\bways\b",
                r"\bprobability\b",
                r"\bstars and bars\b",
                r"\bbinom(?:ial)?\b",
                r"\boutcomes?\b",
                r"\bcombinations?\b",
                r"\bpermutations?\b",
            ),
        ),
        (
            "compute",
            (
                r"\bcalculate\b",
                r"\bcompute\b",
                r"\bevaluate\b",
                r"\bsum\b",
                r"\bproduct\b",
                r"\bdifference\b",
                r"\btotal\b",
                r"\bdivide\b",
                r"\bmultiply\b",
                r"\bsolve\b",
                r"\bfind the midpoint\b",
                r"\bfind the slope\b",
                r"\bfind the area\b",
                r"\bfind the volume\b",
                r"\bfind the equation\b",
                r"=",
            ),
        ),
        (
            "compare",
            (
                r"\bmaximize\b",
                r"\bminimum\b",
                r"\bmaximum\b",
                r"\blargest\b",
                r"\bsmallest\b",
                r"\bcompare\b",
                r"\bat least\b",
                r"\bat most\b",
            ),
        ),
        (
            "conclude",
            (
                r"\btherefore\b",
                r"\bthus\b",
                r"\bhence\b",
                r"\bso\b",
                r"\bwe conclude\b",
                r"\bthe answer is\b",
                r"\bfinal answer\b",
            ),
        ),
    ]
)


SANDWICH_DROP = {"analyze", "rewrite", "apply-formula", "check-constraint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and consolidate pilot reasoning traces into coarse action motifs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def split_reasoning_trace(reasoning_trace: str) -> list[str]:
    cleaned = reasoning_trace.replace("\r\n", "\n")
    cleaned = re.sub(r"\$\$.*?\$\$", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[.*?\\\]", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`{1,3}.*?`{1,3}", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\*+", " ", cleaned)

    raw_units = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9\\])", cleaned)
    units: list[str] = []
    for raw_unit in raw_units:
        unit = raw_unit.strip()
        if not unit:
            continue
        unit = re.sub(r"^\s*(?:[-*]+|\d+\.)\s*", "", unit)
        unit = re.sub(r"\s+", " ", unit).strip()
        if unit:
            units.append(unit)
    return units


def classify_reasoning_unit(unit: str) -> str:
    lowered = unit.lower()
    for token, patterns in CANONICAL_REASONING_TOKENS.items():
        for pattern in patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                return token
    return "analyze"


def collapse_consecutive_duplicates(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        if not out or out[-1] != token:
            out.append(token)
    return out


def collapse_sandwiched_tokens(tokens: list[str]) -> list[str]:
    # Collapse X -> analyze -> X and similar "meta-step" sandwiches.
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            left = tokens[i]
            mid = tokens[i + 1]
            right = tokens[i + 2]
            if left == right and mid in SANDWICH_DROP:
                out.append(left)
                i += 3
                continue
        out.append(tokens[i])
        i += 1
    return out


def collapse_two_token_loops(tokens: list[str]) -> list[str]:
    # Collapse X -> Y -> X -> Y into X -> Y.
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 3 < len(tokens):
            a, b, c, d = tokens[i : i + 4]
            if a == c and b == d:
                out.extend([a, b])
                i += 4
                continue
        out.append(tokens[i])
        i += 1
    return out


def keep_terminal_conclude(tokens: list[str]) -> list[str]:
    if "conclude" not in tokens:
        return tokens
    last_index = len(tokens) - 1 - tokens[::-1].index("conclude")
    out = [token for idx, token in enumerate(tokens) if token != "conclude" or idx == last_index]
    return out


def drop_edge_analyze(tokens: list[str]) -> list[str]:
    out = list(tokens)
    if len(out) > 1 and out[0] == "analyze":
        out = out[1:]
    if len(out) > 1 and out[-1] == "analyze":
        out = out[:-1]
    return out


def consolidate_tokens(tokens: list[str]) -> list[str]:
    current = [token for token in tokens if token]
    if not current:
        return []

    changed = True
    while changed:
        previous = list(current)
        current = collapse_consecutive_duplicates(current)
        current = collapse_sandwiched_tokens(current)
        current = collapse_two_token_loops(current)
        current = keep_terminal_conclude(current)
        current = drop_edge_analyze(current)
        current = collapse_consecutive_duplicates(current)
        changed = current != previous

    return current


def encode_reasoning_trace(reasoning_trace: str) -> str:
    units = split_reasoning_trace(reasoning_trace)
    classified = [classify_reasoning_unit(unit) for unit in units]
    consolidated = consolidate_tokens(classified)
    return "|".join(consolidated)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No rows found in {args.input}")
        fieldnames = list(rows[0].keys()) 
        fieldnames.remove("final_response_text")  # Drop this column since it's not needed anymore.

    if "reasoning_trace" not in fieldnames:
        raise ValueError(f"'reasoning_trace' column not found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["reasoning_trace"] = encode_reasoning_trace(row.get("reasoning_trace", "") or "")
            del output_row["final_response_text"]
            writer.writerow(output_row)

    print(f"Saved {len(rows)} encoded traces to {args.output}")


if __name__ == "__main__":
    main()
