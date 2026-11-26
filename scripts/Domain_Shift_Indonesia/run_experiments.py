#!/usr/bin/env python3
"""Run matched source-only and target fine-tuning experiments sequentially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


CLASSES = {0: "smoke", 1: "fire"}
MODELS = ("yolov9t", "yolov10n", "yolo11n", "yolo26n")
NO_AUG = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "bgr": 0.0,
    "multi_scale": 0.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in paths:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix().encode()
            digest.update(relative)
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def write_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def parse_weights(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in MODELS or not raw_path:
            raise ValueError(f"Bobot harus MODEL=PATH, model: {', '.join(MODELS)}; diterima: {value}")
        if name in parsed:
            raise ValueError(f"Bobot ganda untuk {name}")
        parsed[name] = Path(raw_path).expanduser().resolve()
    if not parsed:
        raise ValueError("Masukkan bobot dengan --weight MODEL=PATH untuk setiap model.")
    return parsed


def check_model(model: YOLO, model_name: str) -> None:
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    observed = {int(key): str(value).lower() for key, value in names.items()}
    if observed != CLASSES:
        raise ValueError(f"Nama kelas {model_name} tidak cocok: diharapkan {CLASSES}, ditemukan {observed}")


def config_for(args: argparse.Namespace, weights: Path, split_hash: str, data_hash: str) -> dict:
    return {
        "weights_sha256": sha256(weights),
        "split_manifest_sha256": split_hash,
        "target_files_sha256": data_hash,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "seed": args.seed,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "device": args.device,
        "workers": args.workers,
        "augmentation": NO_AUG,
        "classes": CLASSES,
    }


def fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def done_if_current(metrics_path: Path, run_id: str) -> bool:
    metrics = read_json(metrics_path)
    if not metrics:
        return False
    if metrics.get("run_id") != run_id:
        raise RuntimeError(
            f"Hasil lama memakai konfigurasi lain: {metrics_path}. Gunakan output baru supaya hasil tidak tercampur."
        )
    return True


def evaluate(weights: Path, data_yaml: Path, destination: Path, args: argparse.Namespace, run_id: str) -> None:
    metrics_path = destination / "metrics.json"
    evaluation_id = fingerprint({
        "training_run_id": run_id,
        "weights_sha256": sha256(weights),
        "data_yaml_sha256": sha256(data_yaml),
    })
    if done_if_current(metrics_path, evaluation_id):
        print(f"[skip] Evaluasi sudah lengkap: {metrics_path}", flush=True)
        return

    model = YOLO(str(weights))
    check_model(model, weights.stem)
    result = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.eval_batch,
        device=args.device,
        workers=args.workers,
        project=str(destination),
        name="plots",
        exist_ok=True,
        plots=True,
        save_txt=True,
        save_conf=True,
    )
    box = result.box
    metrics = {
        "run_id": evaluation_id,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data_yaml": str(data_yaml),
        "split": "test",
        "imgsz": args.imgsz,
        "ultralytics_version": ultralytics.__version__,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": {
            str(result.names[int(class_id)]): {
                "ap50": float(box.ap50[index]),
                "ap50_95": float(box.ap[index]),
                "precision": float(box.p[index]),
                "recall": float(box.r[index]),
            }
            for index, class_id in enumerate(box.ap_class_index)
        },
    }
    (destination / "plots" / "labels").mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def adapt(weights: Path, data_yaml: Path, destination: Path, args: argparse.Namespace, config: dict, run_id: str) -> Path:
    training_dir = destination / "training"
    best = training_dir / "weights" / "best.pt"
    config_path = training_dir / "run_config.json"
    last = training_dir / "weights" / "last.pt"

    if best.is_file():
        saved = read_json(config_path)
        if saved.get("run_id") != run_id:
            raise RuntimeError(
                f"Checkpoint adaptation dibuat dengan konfigurasi berbeda: {best}. Pilih folder output baru."
            )
        print(f"[skip] Training sudah lengkap: {best}", flush=True)
        return best

    training_dir.mkdir(parents=True, exist_ok=True)
    saved = read_json(config_path)
    if last.is_file() and saved.get("run_id") != run_id:
        raise RuntimeError(
            f"Checkpoint parsial dibuat dengan konfigurasi berbeda: {last}. Gunakan folder output baru."
        )
    write_json(config_path, {"run_id": run_id, "config": config, "started_utc": saved.get("started_utc", datetime.now(timezone.utc).isoformat())})
    start = {"data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
             "optimizer": args.optimizer, "lr0": args.lr0, "seed": args.seed, "device": args.device,
             "workers": args.workers}
    if last.is_file():
        print(f"[resume] Melanjutkan checkpoint terakhir: {last}", flush=True)
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(str(weights))
        check_model(model, weights.stem)
        model.train(
            **start,
            patience=args.patience,
            deterministic=True,
            project=str(destination),
            name="training",
            exist_ok=True,
            plots=True,
            save=True,
            **NO_AUG,
        )
    if not best.is_file():
        raise FileNotFoundError(f"Training tidak menghasilkan best.pt: {best}")
    saved = read_json(config_path)
    saved["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, saved)
    return best


def summarize(output: Path, weights: dict[str, Path]) -> None:
    rows = []
    for model in weights:
        locations = {
            "full_target_exploratory": output / "results_full" / model / "metrics.json",
            "test_before": output / "results_adapt" / model / "before" / "metrics.json",
            "test_after": output / "results_adapt" / model / "after" / "metrics.json",
        }
        for stage, path in locations.items():
            if not path.is_file():
                continue
            metrics = read_json(path)
            rows.append({"model": model, "stage": stage, **{key: metrics[key] for key in ("precision", "recall", "map50", "map50_95")}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ringkasan_tiga_evaluasi.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "stage", "precision", "recall", "map50", "map50_95"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Folder berisi full/ dan split/ hasil persiapan notebook.")
    parser.add_argument("--output", type=Path, required=True, help="Folder output run yang baru atau dapat dilanjutkan.")
    parser.add_argument("--weight", action="append", default=[], help="MODEL=PATH, dapat diulang untuk empat model.")
    parser.add_argument("--stage", choices=("all", "evaluate-before", "adapt", "evaluate-after"), default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    prepared = args.prepared.expanduser().resolve()
    output = args.output.expanduser().resolve()
    weights = parse_weights(args.weight)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Aktifkan GPU pada Kaggle/HPC sebelum training.")
    for item in (prepared / "full/data.yaml", prepared / "split/data.yaml", prepared / "split_manifest.csv"):
        if not item.is_file():
            raise FileNotFoundError(f"File split wajib tidak ditemukan: {item}")
    for model, path in weights.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {model} tidak ditemukan: {path}")

    split_hash = sha256(prepared / "split_manifest.csv")
    data_hash = tree_hash([prepared / "split/images", prepared / "split/labels", prepared / "full/images/test", prepared / "full/labels/test"])
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = output / "split_manifest.csv"
    if previous_manifest.is_file() and sha256(previous_manifest) != split_hash:
        raise RuntimeError("Folder output memiliki split berbeda. Gunakan nama output baru agar hasil tidak tercampur.")
    shutil.copy2(prepared / "split_manifest.csv", output / "split_manifest.csv")
    test_labels = prepared / "split/labels/test"
    ground_truth = output / "ground_truth"
    ground_truth.mkdir(exist_ok=True)
    for label in test_labels.glob("*.txt"):
        shutil.copy2(label, ground_truth / label.name)

    print(f"GPU: {torch.cuda.get_device_name(0)} | Ultralytics: {ultralytics.__version__}", flush=True)
    print(f"Split SHA-256: {split_hash} | Data SHA-256: {data_hash} | Output: {output}", flush=True)
    for model_name, source_weights in weights.items():
        config = config_for(args, source_weights, split_hash, data_hash)
        run_id = fingerprint(config)
        model_output = output / "results_adapt" / model_name
        full_output = output / "results_full" / model_name
        write_json(model_output / "run_config.json", {"run_id": run_id, "model": model_name, "config": config})

        if args.stage in ("all", "evaluate-before"):
            evaluate(source_weights, prepared / "full/data.yaml", full_output, args, run_id)
            evaluate(source_weights, prepared / "split/data.yaml", model_output / "before", args, run_id)
        if args.stage in ("all", "adapt"):
            best = adapt(source_weights, prepared / "split/data.yaml", model_output, args, config, run_id)
        else:
            best = model_output / "training/weights/best.pt"
        if args.stage in ("all", "evaluate-after"):
            if not best.is_file():
                raise FileNotFoundError(f"Checkpoint adaptation belum ada untuk {model_name}: {best}")
            evaluate(best, prepared / "split/data.yaml", model_output / "after", args, run_id)
        print(f"[selesai] {model_name}", flush=True)

    summarize(output, weights)
    print(f"Ringkasan tersimpan: {output / 'ringkasan_tiga_evaluasi.csv'}", flush=True)


if __name__ == "__main__":
    main()
