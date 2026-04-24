import argparse
import csv
from pathlib import Path


def clean_file(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = input_path.with_name(f"clean_{input_path.name}")

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("Input CSV has no header row.")
        if "tokenized_trace" not in fieldnames:
            raise ValueError("Input CSV is missing required column: tokenized_trace")

        kept_rows = []
        total_rows = 0
        for row in reader:
            total_rows += 1
            if row.get("tokenized_trace") != "MISSING":
                kept_rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    removed_rows = total_rows - len(kept_rows)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows read: {total_rows}")
    print(f"Rows removed (tokenized_trace == MISSING): {removed_rows}")
    print(f"Rows written: {len(kept_rows)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rows where tokenized_trace is MISSING."
    )
    parser.add_argument(
        "input_name",
        help="Input CSV filename (or path). Output will be clean_{input_name}.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_name)
    clean_file(input_path)


if __name__ == "__main__":
    main()
