from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "pilot_traces.csv"
OUTPUT_PATH = ROOT / "tokenized_pilot_traces.csv"


# Coarse canonical reasoning actions inferred from the pilot traces.
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


def compress_tokens(tokens: list[str]) -> list[str]:
    compressed: list[str] = []
    for token in tokens:
        if not compressed or compressed[-1] != token:
            compressed.append(token)
    return compressed


def encode_reasoning_trace(reasoning_trace: str) -> str:
    units = split_reasoning_trace(reasoning_trace)
    tokens = [classify_reasoning_unit(unit) for unit in units]
    return "|".join(compress_tokens(tokens))


def main() -> None:
    with INPUT_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No rows found in {INPUT_PATH}")
        fieldnames = list(rows[0].keys())
        fieldnames.remove("final_response_text") 

    if "reasoning_trace" not in fieldnames:
        raise ValueError(f"'reasoning_trace' column not found in {INPUT_PATH}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["reasoning_trace"] = encode_reasoning_trace(row.get("reasoning_trace", "") or "")
            del output_row["final_response_text"]
            writer.writerow(output_row)

    print(f"Saved {len(rows)} encoded pilot traces to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
