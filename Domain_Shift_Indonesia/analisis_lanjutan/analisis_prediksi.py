"""Hitung TP/FP/FN untuk prediksi tersimpan pada threshold yang dipilih."""

import argparse
import csv
from pathlib import Path

from bbox_analysis import MODELS, dataset_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("confidence dan iou harus di antara 0 dan 1")

    root = Path(__file__).resolve().parent.parent / "arsip" / "hasil_yolo11n_26n_20261123"
    rows = []
    for model in args.model or MODELS:
        for stage in ("before", "after"):
            rows.extend(dataset_audit(root, model, stage, args.confidence, args.iou))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ditulis {len(rows)} baris ke {args.output}")


if __name__ == "__main__":
    main()
