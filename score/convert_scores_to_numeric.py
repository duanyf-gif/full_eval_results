from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "results" / "results_itfi_layer23_top50_lam2p0_random.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "results" / "results_itfi_layer23_top50_lam2p0_random_numeric.csv"
SCORE_COLUMNS = [
    "brushstroke",
    "color",
    "composition",
    "light_and_shadow",
    "line_quality",
]

SCORE_MAPPING: Dict[str, str] = {
    "excellent": "5",
    "good": "4",
    "fair": "3",
    "poor": "2",
    "very poor": "1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert qualitative score labels in results.csv to numeric scores."
    )
    parser.add_argument(
        "--input_csv",
        default=str(DEFAULT_INPUT_CSV),
        help=f"Input CSV path. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output_csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    return parser


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def map_score(value: str) -> str:
    normalized = normalize_label(value)
    return SCORE_MAPPING.get(normalized, "")


def convert_scores(input_csv: Path, output_csv: Path) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    converted_rows = 0
    score_sums = {column: 0.0 for column in SCORE_COLUMNS}
    score_counts = {column: 0 for column in SCORE_COLUMNS}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")

        with output_csv.open("w", encoding="utf-8-sig", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                converted_row = dict(row)
                for column in SCORE_COLUMNS:
                    if column in converted_row:
                        converted_row[column] = map_score(converted_row[column])
                        if converted_row[column]:
                            score_sums[column] += float(converted_row[column])
                            score_counts[column] += 1
                writer.writerow(converted_row)
                converted_rows += 1

            average_row = {field: "" for field in fieldnames}
            average_row["image_name"] = "AVERAGE"
            for column in SCORE_COLUMNS:
                if column in average_row and score_counts[column] > 0:
                    average_row[column] = f"{score_sums[column] / score_counts[column]:.4f}"
            writer.writerow(average_row)

    return converted_rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    converted_rows = convert_scores(input_csv=input_csv, output_csv=output_csv)
    print(f"Converted {converted_rows} row(s).")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
