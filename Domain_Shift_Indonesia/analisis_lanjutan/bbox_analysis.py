"""Analisis prediksi bbox pada satu split; ini bukan implementasi AP resmi."""

from pathlib import Path

CLASSES = {0: "smoke", 1: "fire"}
MODELS = ("yolov9t", "yolov10n", "yolo11n", "yolo26n")


def read_boxes(path, prediction=False):
    if not path.exists():
        return []
    boxes = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != (6 if prediction else 5):
            raise ValueError(f"Format bbox tidak valid: {path}:{number}")
        cls = int(values[0])
        if cls not in CLASSES:
            raise ValueError(f"Kelas tidak dikenal: {path}:{number}: {cls}")
        x, y, width, height = map(float, values[1:5])
        minimum_size = 0 if prediction else 1e-12
        if not (0 <= x <= 1 and 0 <= y <= 1 and minimum_size <= width <= 1 and minimum_size <= height <= 1):
            raise ValueError(f"Koordinat bbox tidak valid: {path}:{number}")
        boxes.append({"class_id": cls, "xywh": (x, y, width, height),
                      "confidence": float(values[5]) if prediction else None})
    return boxes


def box_iou(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    intersection = max(0., min(x + w / 2, xx + ww / 2) - max(x - w / 2, xx - ww / 2))
    intersection *= max(0., min(y + h / 2, yy + hh / 2) - max(y - h / 2, yy - hh / 2))
    union = w * h + ww * hh - intersection
    return intersection / union if union > 0 else 0.


def match_boxes(ground_truth, predictions, class_id, confidence=0.25, threshold=0.5):
    """Match one GT to at most one confidence-ordered prediction of the same class."""
    gt = [(i, item) for i, item in enumerate(ground_truth) if item["class_id"] == class_id]
    pred = sorted(((i, item) for i, item in enumerate(predictions)
                   if item["class_id"] == class_id and item["confidence"] >= confidence),
                  key=lambda pair: -pair[1]["confidence"])
    used, matches, false_positives = set(), [], []
    for pred_index, item in pred:
        candidates = [(box_iou(item["xywh"], target["xywh"]), gt_index)
                      for gt_index, target in gt if gt_index not in used]
        best_iou, gt_index = max(candidates, key=lambda pair: pair[0], default=(0., None))
        if gt_index is not None and best_iou >= threshold:
            used.add(gt_index)
            matches.append((gt_index, pred_index, best_iou))
        else:
            false_positives.append(pred_index)
    return matches, false_positives, [i for i, _ in gt if i not in used]


def audit_case(ground_truth, predictions, class_id, confidence=0.25, threshold=0.5):
    matches, fp, fn = match_boxes(ground_truth, predictions, class_id, confidence, threshold)
    return {"tp": len(matches), "fp": len(fp), "fn": len(fn),
            "matches": matches, "fp_indices": fp, "fn_indices": fn}


def fn_indicators(ground_truth, predictions, class_id, confidence=0.25, threshold=0.5):
    """Heuristic indicators for unmatched GT; candidates can be shared across GTs."""
    result = audit_case(ground_truth, predictions, class_id, confidence, threshold)
    indicators = []
    for index in result["fn_indices"]:
        target = ground_truth[index]
        overlaps = [(box_iou(target["xywh"], p["xywh"]), p)
                    for p in predictions if p["class_id"] == class_id]
        high = max((iou for iou, p in overlaps if p["confidence"] >= confidence), default=0.)
        low = max((iou for iou, p in overlaps if p["confidence"] < confidence), default=0.)
        other = max((box_iou(target["xywh"], p["xywh"]) for p in predictions
                     if p["class_id"] != class_id and p["confidence"] >= confidence), default=0.)
        if high >= threshold:
            category = "kandidat sudah dipasangkan"
        elif low >= threshold:
            category = "kandidat di bawah confidence"
        elif other >= threshold:
            category = "kemungkinan kelas berbeda"
        elif high >= 0.1:
            category = "overlap parsial (0,1–0,5)"
        else:
            category = "tidak ada prediksi terdekat"
        indicators.append({"gt_index": index, "category": category,
                           "best_iou_above_conf": high, "best_iou_below_conf": low,
                           "best_iou_other_class": other})
    return indicators


def dataset_audit(root, model, stage, confidence=0.25, threshold=0.5):
    root = Path(root)
    labels = root / "ground_truth"
    if not labels.is_dir():
        labels = root / "prepared" / "split" / "labels" / "test"
    predictions = root / "results_adapt" / model / stage / "plots" / "labels"
    if not predictions.is_dir():
        predictions = root / "run_data" / model / stage / "plots" / "labels"
    if not predictions.is_dir():
        raise FileNotFoundError(f"Folder prediksi tidak tersedia: {predictions}")
    rows = []
    for label in sorted(labels.glob("*.txt")):
        gt = read_boxes(label)
        pred = read_boxes(predictions / label.name, prediction=True)
        for class_id, name in CLASSES.items():
            result = audit_case(gt, pred, class_id, confidence, threshold)
            rows.append({"model": model, "stage": stage, "image_stem": label.stem,
                         "class": name, "tp": result["tp"], "fp": result["fp"], "fn": result["fn"]})
    return rows
