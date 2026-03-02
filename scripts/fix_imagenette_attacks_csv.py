#!/usr/bin/env python3
"""
Fix polluted clean_accuracy, accuracy_drop, relative_accuracy_drop in imagenette_attacks_lab CSV files.
The bug: clean_accuracy was passed as 0-100 but used as 0-1 in formulas.
"""
import csv
import os

RESULTS_DIR = "results/imagenette_attacks_lab"


def fix_row(row: dict) -> dict:
    """Fix effectiveness metrics in a row."""
    try:
        clean = float(row.get("clean_accuracy", 0))
        acc = float(row.get("acc", 0))
    except (ValueError, TypeError):
        return row

    # Normalize clean_accuracy to 0-1
    clean_ratio = clean / 100.0 if clean > 1.0 else clean
    robust_ratio = acc / 100.0

    # Recompute accuracy_drop and relative_accuracy_drop
    accuracy_drop = max(0.0, clean_ratio - robust_ratio)
    relative_accuracy_drop = accuracy_drop / clean_ratio if clean_ratio > 0 else 0.0

    row["clean_accuracy"] = f"{clean_ratio:.10g}"
    row["accuracy_drop"] = f"{accuracy_drop:.10g}"
    row["relative_accuracy_drop"] = f"{relative_accuracy_drop:.10g}"
    return row


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Directory not found: {RESULTS_DIR}")
        return

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        if not rows or "clean_accuracy" not in fieldnames:
            continue

        fixed_rows = [fix_row(row) for row in rows]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fixed_rows)

        print(f"Fixed: {filepath}")


if __name__ == "__main__":
    main()
