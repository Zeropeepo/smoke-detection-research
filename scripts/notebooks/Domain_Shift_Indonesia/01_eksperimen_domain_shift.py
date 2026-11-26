# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluasi domain Indonesia dan supervised fine-tuning — YOLO
# Alur: (1) source-only seluruh target; (2) source-only test; (3) fine-tuning train target, pilih best pada val, evaluasi test yang sama. Eksperimen berurutan. Hasil bersifat eksploratif karena label, sumber, duplikat, dan independensi kejadian belum selesai diaudit. Penggunaan istilah zero-shot di sini berarti **source-only**, bukan open-vocabulary.
#
# Semua checkpoint awal harus berasal dari Smoke and Fire dan memiliki kelas 0=smoke, 1=fire. Runner melatih model berurutan dengan augmentasi target yang sama-sama dimatikan, seed/split/hyperparameter yang sama, lalu menyimpan hash untuk audit. Riwayat training source checkpoint tetap berbeda, sehingga hasil adalah perbandingan transfer bobot yang tersedia. Jangan gunakan test untuk memilih threshold atau hyperparameter.

# %% [markdown]
# ## 1. Runtime dan pemeriksaan GPU
#
# Notebook memakai paket yang sudah tersedia di Kaggle/HPC dan tidak memasang library. Catat versi Ultralytics dan gunakan versi yang sama untuk seluruh run.

# %%
import ultralytics
print('Ultralytics:', ultralytics.__version__)

# %%
import sys, os, json, csv, hashlib, collections, random, shutil, zipfile
from pathlib import Path
from types import SimpleNamespace
import torch, ultralytics
from ultralytics import YOLO
from IPython.display import display
import pandas as pd

print('Python:', sys.version.split()[0])
print('Ultralytics:', ultralytics.__version__)
print('Torch:', torch.__version__)
print('CUDA tersedia:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Aktifkan GPU pada pengaturan Kaggle sebelum melanjutkan.'
print('GPU:', torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Data, checkpoint, dan konfigurasi
# Tambahkan dataset privat ke Kaggle. Ubah path checkpoint saat sudah tersedia dan pilih model yang ingin dijalankan. Semua checkpoint wajib mendeteksi `0: smoke`, `1: fire`. Jangan gunakan bobot bawaan COCO untuk menggantikan checkpoint Smoke and Fire. `REUSE_PREPARED` opsional hanya jika sudah memastikan split dan label sama persis.

# %%
INPUT_ROOT = Path('/kaggle/input/datasets/windy25/karhutla-yolov9t-experiment-inputs')
DATASET_DIR = INPUT_ROOT / 'combined_manual_review' / 'combined_manual_review'
# Sesuaikan semua path checkpoint jika nama/path Kaggle input berbeda.
WEIGHTS_ALL = {
    'yolov9t': INPUT_ROOT / 'yolov9t_smoke_fire_best.pt',
    'yolov10n': INPUT_ROOT / 'yolov10n_smoke_fire_best.pt',
    'yolo11n': Path('/kaggle/input/datasets/windy25/karhutla-source-yolo11n-yolo26n-checkpoints/yolo11n_source_best.pt'),
    'yolo26n': Path('/kaggle/input/datasets/windy25/karhutla-source-yolo11n-yolo26n-checkpoints/yolo26n_source_best.pt'),
}
SELECTED_MODELS = ['yolov9t', 'yolov10n', 'yolo11n', 'yolo26n']
WEIGHTS = {name: WEIGHTS_ALL[name] for name in SELECTED_MODELS}
RUN_DIR = Path('/kaggle/working/karhutla_matched_aug_v1')
existing_prepared = RUN_DIR / 'prepared'
REUSE_PREPARED = existing_prepared if all((existing_prepared / item).is_file() for item in ('full/data.yaml', 'split/data.yaml', 'split_manifest.csv', 'split_report.json')) else None
PREPARED = Path(REUSE_PREPARED) if REUSE_PREPARED else existing_prepared
SEED, IMGSZ, EVAL_BATCH, TRAIN_BATCH = 42, 640, 8, 16
EPOCHS, PATIENCE, LR0, WORKERS, DEVICE = 50, 15, 0.001, 2, '0'
ANALYSIS_CONF, ANALYSIS_IOU = 0.25, 0.50  # deskriptif; pilih threshold operasional di validation.
print('Output:', RUN_DIR, '\nModel:', WEIGHTS)

# %%
if not DATASET_DIR.is_dir():
    archive = INPUT_ROOT / 'combined_manual_review.zip'
    if not archive.is_file():
        raise FileNotFoundError(f'Dataset atau ZIP tidak ditemukan di {INPUT_ROOT}; pastikan input ditambahkan ke notebook.')
    extracted = RUN_DIR / 'source'
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Arsip privat buatan sendiri: tetap periksa path agar tidak keluar dari folder ekstraksi.
        for member in z.infolist():
            target = (extracted / member.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError(f'Path tidak aman dalam ZIP: {member.filename}')
        z.extractall(extracted)
    DATASET_DIR = extracted / 'combined_manual_review'

assert (DATASET_DIR / 'manifest.csv').is_file(), f'manifest.csv tidak ditemukan: {DATASET_DIR}'
assert (DATASET_DIR / 'labels').is_dir(), 'Folder labels tidak ditemukan'
for name, path in WEIGHTS.items():
    assert path.is_file(), f'Checkpoint {name} tidak ditemukan: {path}. Periksa apakah dataset Kaggle sudah menggunakan versi terbaru.'
print('Data dan checkpoint ditemukan.')

# %% [markdown]
# ## 3. Fungsi persiapan split
# Split mempertahankan grup yang memiliki video, URL, atau hash gambar yang sama; data tanpa sumber akan dikelompokkan per gambar. Kamu bisa meninjau dan mengubah fungsi di bawah sebelum menjalankannya.

# %%
import argparse
RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
CLASSES = {0: 'smoke', 1: 'fire'}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def prepare(args):
    source = args.dataset.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output tidak kosong: {out}. Pilih folder output baru.")
    rows = list(csv.DictReader((source / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    images = {p.name: p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")}
    if len(rows) != len(images) or set(images) != {r["image_filename"] for r in rows}:
        raise ValueError("Manifest dan file gambar tidak cocok; periksa nama/duplikasi.")
    override = {}
    if args.groups:
        with args.groups.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["image_filename"] not in images or not row["group_id"].strip():
                    raise ValueError(f"Mapping grup tidak valid: {row}")
                if row["image_filename"] in override:
                    raise ValueError(f"Mapping grup duplikat: {row['image_filename']}")
                override[row["image_filename"]] = row["group_id"].strip()
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    seen, unknown, classes = {}, [], []
    for i, row in enumerate(rows):
        name = row["image_filename"]
        im = images[name]
        label = source / "labels" / (im.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Label tidak ditemukan: {label}")
        found = set()
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Label tidak valid: {label}: {line}")
            c, x, y, w, h = map(float, parts)
            if c not in CLASSES or w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                raise ValueError(f"Bounding box tidak valid: {label}: {line}")
            found.add(int(c))
        classes.append(found)
        explicit = override.get(name)
        video = row.get("video_id", "").strip()
        url = row.get("source_url", "").strip()
        if not explicit and not video and not url:
            unknown.append(name)
        # Joining by all known identifiers and image hash protects shared source and exact duplicates.
        keys = [("hash", sha(im))]
        if explicit: keys.append(("event", explicit))
        if video: keys.append(("video", video))
        if url: keys.append(("url", url))
        for key in keys:
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    if unknown and not args.allow_unknown:
        raise ValueError(f"{len(unknown)} gambar tidak memiliki sumber/grup. Lengkapi --groups atau gunakan --allow-unknown untuk split SEMENTARA.")
    groups = collections.defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    group_list = list(groups.values())
    total = [len(rows), sum(0 in c for c in classes), sum(1 in c for c in classes), len(group_list)]
    rng = random.Random(args.seed)
    best = None
    for _ in range(1200):
        order = group_list[:]
        rng.shuffle(order)
        order.sort(key=len, reverse=True)
        counts = {s: [0, 0, 0, 0] for s in RATIOS}
        assignment = {}
        for group in order:
            vec = [len(group), sum(0 in classes[i] for i in group), sum(1 in classes[i] for i in group), 1]
            def score(split):
                return sum(((counts[s][j] + (vec[j] if s == split else 0) - total[j]*RATIOS[s]) / max(total[j]*RATIOS[s], 1))**2 for s in RATIOS for j in range(4))
            split = min(RATIOS, key=score)
            assignment.update({i: split for i in group})
            counts[split] = [a+b for a, b in zip(counts[split], vec)]
        final = sum(((counts[s][j]-total[j]*RATIOS[s])/max(total[j]*RATIOS[s],1))**2 for s in RATIOS for j in range(4))
        if best is None or final < best[0]: best = (final, assignment, counts)
    _, assignment, counts = best
    # Refine whole-group moves; the greedy pass alone can leave avoidable split imbalance.
    def objective():
        return sum(((counts[s][j]-total[j]*RATIOS[s])/max(total[j]*RATIOS[s],1))**2 for s in RATIOS for j in range(4))
    for _ in range(30000):
        group = rng.choice(group_list)
        old = assignment[group[0]]
        new = rng.choice([s for s in RATIOS if s != old])
        vec = [len(group), sum(0 in classes[i] for i in group), sum(1 in classes[i] for i in group), 1]
        previous = objective()
        counts[old] = [a-b for a, b in zip(counts[old], vec)]
        counts[new] = [a+b for a, b in zip(counts[new], vec)]
        if objective() < previous:
            assignment.update({i: new for i in group})
        else:
            counts[old] = [a+b for a, b in zip(counts[old], vec)]
            counts[new] = [a-b for a, b in zip(counts[new], vec)]
    # Copy, never alter the original annotations. Full set duplicates are retained intentionally.
    for mode in ("full", "split"):
        for i, row in enumerate(rows):
            im = images[row["image_filename"]]
            split = "test" if mode == "full" else assignment[i]
            for sub, src in (("images", im), ("labels", source / "labels" / (im.stem + ".txt"))):
                dest = out / mode / sub / split / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        base = out / mode
        payload = {"path": str(base), "train": "images/train" if mode == "split" else "images/test", "val": "images/val" if mode == "split" else "images/test", "test": "images/test", "names": CLASSES}
        write_json(base / "data.yaml", payload)  # JSON is a YAML subset.
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "split", "group", "source_known", "sha256_image", "classes"])
        writer.writeheader()
        for i, row in enumerate(rows):
            name = row["image_filename"]
            writer.writerow({"image_filename": name, "split": assignment[i], "group": f"group_{find(i):03d}", "source_known": name not in unknown, "sha256_image": sha(images[name]), "classes": ",".join(CLASSES[c] for c in sorted(classes[i]))})
    stats = {s: {"images": counts[s][0], "smoke_images": counts[s][1], "fire_images": counts[s][2], "groups": len({find(i) for i in assignment if assignment[i] == s})} for s in RATIOS}
    report = {"seed": args.seed, "ratios_target": RATIOS, "images": len(rows), "groups": len(groups), "unknown_source_count": len(unknown), "unknown_source_images": unknown, "provisional": bool(unknown), "split": stats, "warning": "Exact duplicates kept and grouped; near-duplicates, unknown events and inconsistent labels need manual review."}
    write_json(out / "split_report.json", report)
    print(json.dumps(report | {"unknown_source_images": f"{len(unknown)} filenames in split_report.json"}, indent=2))

# %% [markdown]
# ## 4. Siapkan data dan tinjau distribusi
# Jika `REUSE_PREPARED` disetel, notebook menggunakan split itu tanpa mengubahnya. Jika tidak, split baru dibuat. Split berdasarkan video, URL, dan hash yang diketahui; split tidak dapat menjamin pemisahan kejadian jika sumber tidak diketahui.

# %%
if REUSE_PREPARED:
    for required in ('full/data.yaml', 'split/data.yaml', 'split_manifest.csv', 'split_report.json'):
        assert (PREPARED / required).is_file(), f'Tidak ditemukan: {PREPARED / required}'
else:
    prepare(SimpleNamespace(dataset=DATASET_DIR, output=PREPARED, groups=None, allow_unknown=True, seed=SEED))
report = json.loads((PREPARED / 'split_report.json').read_text())
display(pd.DataFrame(report['split']).T)
print('Sumber belum diketahui:', report['unknown_source_count'])
print('Split sementara:', report['provisional'])
print('SHA-256 split manifest:', sha(PREPARED / 'split_manifest.csv'))

# %% [markdown]
# ## 5. Fungsi evaluasi
# Hasil mencakup precision, recall, mAP@50, mAP@50–95, dan AP per kelas; juga grafik dan prediksi teks.

# %%
def check_model(model):
    names = model.names
    if isinstance(names, list): names = dict(enumerate(names))
    if {int(k): str(v).lower() for k, v in names.items()} != CLASSES:
        raise ValueError(f"Kelas checkpoint tidak cocok (diharapkan {CLASSES}): {names}")

def evaluate(YOLO, weights, data, dest, args, version):
    model = YOLO(str(weights))
    check_model(model)
    dest.mkdir(parents=True, exist_ok=True)
    m = model.val(data=str(data), split="test", imgsz=args.imgsz, batch=args.eval_batch, device=args.device,
                  workers=args.workers, project=str(dest), name="plots", exist_ok=False, plots=True, save_txt=True, save_conf=True)
    (dest / "plots" / "labels").mkdir(parents=True, exist_ok=True)  # no predictions is valid
    box = m.box
    values = {"weights": str(weights.resolve()), "weights_sha256": sha(weights), "dataset_yaml": str(data.resolve()), "split": "test", "imgsz": args.imgsz, "ultralytics_version": version, "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map), "per_class": {str(m.names[int(c)]): {"ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]), "precision": float(box.p[i]), "recall": float(box.r[i])} for i, c in enumerate(box.ap_class_index)}}
    write_json(dest / "metrics.json", values)
    print(json.dumps(values, indent=2))

# %%
eval_args = SimpleNamespace(imgsz=IMGSZ, eval_batch=EVAL_BATCH, device=DEVICE, workers=WORKERS)

def run_evaluation(weights, data_yaml, destination):
    return evaluate(YOLO, Path(weights), Path(data_yaml), Path(destination), eval_args, ultralytics.__version__)

# %% [markdown]
# ## 6. Jalankan pipeline berpasangan untuk empat model
#
# Runner menjalankan evaluasi source-only, fine-tuning target, dan evaluasi sesudah fine-tuning secara berurutan. Semua model memakai split dan augmentasi target yang sama. Untuk melanjutkan run yang terputus, jalankan ulang sel ini dengan folder output yang sama; tahap lengkap akan dilewati setelah hash konfigurasi cocok. Jika data, checkpoint, atau konfigurasi berubah, gunakan folder output baru.
#
# Tambahkan `run_experiments.py` sebagai Kaggle Dataset input, lalu atur `KARHUTLA_RUNNER` ke lokasi file tersebut. Pastikan empat checkpoint sumber Smoke and Fire tersedia di input Kaggle.

# %%
import subprocess

RUNNER_PATH = Path(os.environ.get(
    "KARHUTLA_RUNNER",
    "/kaggle/input/datasets/windy25/karhutla-experiment-runner/run_experiments.py",
))
if not RUNNER_PATH.is_file():
    raise FileNotFoundError(
        f"Runner tidak ditemukan: {RUNNER_PATH}. Tambahkan scripts/Domain_Shift_Indonesia/run_experiments.py "
        "sebagai Kaggle input dan set KARHUTLA_RUNNER ke path aktualnya."
    )

command = [
    sys.executable, str(RUNNER_PATH),
    "--prepared", str(PREPARED),
    "--output", str(RUN_DIR),
    "--imgsz", str(IMGSZ),
    "--epochs", str(EPOCHS),
    "--patience", str(PATIENCE),
    "--batch", str(TRAIN_BATCH),
    "--eval-batch", str(EVAL_BATCH),
    "--optimizer", "SGD",
    "--lr0", str(LR0),
    "--seed", str(SEED),
    "--device", DEVICE,
    "--workers", str(WORKERS),
]
for model_name, checkpoint in WEIGHTS.items():
    command.extend(["--weight", f"{model_name}={checkpoint}"])
subprocess.run(command, check=True)
BEST = {name: RUN_DIR / "results_adapt" / name / "training/weights/best.pt" for name in WEIGHTS}

# %% [markdown]
# ## 7. Ringkasan hasil
# Angka ditampilkan dalam persen. Selisih menunjukkan after dikurangi before pada test Indonesia yang sama.

# %%
rows = []
for name in WEIGHTS:
    paths = {
        'Full 376 (eksploratif)': RUN_DIR / 'results_full' / name / 'metrics.json',
        'Test sebelum adaptation': RUN_DIR / 'results_adapt' / name / 'before/metrics.json',
        'Test setelah adaptation': RUN_DIR / 'results_adapt' / name / 'after/metrics.json',
    }
    for label, path in paths.items():
        if path.exists():
            m = json.loads(path.read_text())
            rows.append({'Model': name, 'Kondisi': label, **{key: round(100*m[key], 2) for key in ('precision', 'recall', 'map50', 'map50_95')}})
df = pd.DataFrame(rows)
display(df)
RUN_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RUN_DIR / "ringkasan_tiga_evaluasi.csv", index=False)
if not df.empty:
    metrics = ['precision', 'recall', 'map50', 'map50_95']
    for name in WEIGHTS:
        subset = df[df['Model'] == name].set_index('Kondisi')
        if 'Test sebelum adaptation' in subset.index and 'Test setelah adaptation' in subset.index:
            print(f'{name}: selisih setelah − sebelum (poin persentase)')
            display((subset.loc['Test setelah adaptation', metrics] - subset.loc['Test sebelum adaptation', metrics]).to_frame('Δ pp'))
print('Seluruh output:', RUN_DIR)
print('Simpan versi notebook/output di Kaggle untuk mempertahankan checkpoint dan hasil.')

# %% [markdown]
# ## 8. Analisis kesalahan per kelas dan gambar
# Penghitungan di bawah memakai confidence dan IoU tetap; ini bukan AP dan tidak harus identik dengan evaluator resmi. Prediksi salah terhadap anotasi perlu ditinjau manual, bukan otomatis dianggap kesalahan model. Hindari memilih model berdasarkan test eksploratif.

# %%
from pathlib import Path
from collections import Counter
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Image as IPImage

# 1. LOKASI HASIL DAN PENGATURAN ANALISIS

RUN_DIR = Path(RUN_DIR)

TEST_IMAGES = PREPARED / "split/images/test"
TEST_LABELS = PREPARED / "split/labels/test"
OUTPUT_DIR = RUN_DIR / "analisis_lanjutan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = list(WEIGHTS)
CLASS_NAMES = {0: "smoke", 1: "fire"}

# Digunakan hanya untuk analisis TP/FP/FN di bawah.
# Ini bukan threshold yang otomatis dipakai untuk menghitung mAP.
CONF_THRESHOLD = ANALYSIS_CONF
IOU_THRESHOLD = ANALYSIS_IOU

assert RUN_DIR.is_dir(), f"Folder hasil tidak ditemukan: {RUN_DIR}"
assert TEST_IMAGES.is_dir(), f"Folder gambar test tidak ditemukan: {TEST_IMAGES}"
assert TEST_LABELS.is_dir(), f"Folder label test tidak ditemukan: {TEST_LABELS}"

print("Folder hasil:", RUN_DIR)
print("Folder analisis:", OUTPUT_DIR)
print("Confidence threshold analisis:", CONF_THRESHOLD)
print("IoU threshold analisis:", IOU_THRESHOLD)


# 2. FUNGSI PEMBANTU

def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with path.open("r") as file:
        return json.load(file)


def load_yolo_boxes(path, is_prediction=False):
    """
    Ground truth:
        class x_center y_center width height

    Prediction:
        class x_center y_center width height confidence

    Jika file prediksi tidak ada, dianggap tidak ada deteksi.
    Folder prediksi tetap diperiksa terlebih dahulu.
    """
    path = Path(path)

    if not path.exists():
        return []

    boxes = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        expected = 6 if is_prediction else 5

        if len(parts) != expected:
            raise ValueError(
                f"Format tidak sesuai di {path}, baris {line_number}: {line}"
            )

        class_id = int(float(parts[0]))
        x, y, width, height = map(float, parts[1:5])

        confidence = float(parts[5]) if is_prediction else None

        boxes.append({
            "class_id": class_id,
            "xywh": (x, y, width, height),
            "confidence": confidence,
        })

    return boxes


def box_iou(box_a, box_b):
    """
    Menghitung IoU dua bbox YOLO yang dinormalisasi:
    x_center, y_center, width, height.
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    a_left = xa - wa / 2
    a_top = ya - ha / 2
    a_right = xa + wa / 2
    a_bottom = ya + ha / 2

    b_left = xb - wb / 2
    b_top = yb - hb / 2
    b_right = xb + wb / 2
    b_bottom = yb + hb / 2

    inter_width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    inter_height = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    intersection = inter_width * inter_height

    area_a = max(0.0, wa) * max(0.0, ha)
    area_b = max(0.0, wb) * max(0.0, hb)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(ground_truth, predictions, class_id):
    """
    Pencocokan sederhana per kelas:
    - prediksi diurutkan berdasarkan confidence;
    - satu prediksi hanya bisa dipasangkan dengan satu ground truth;
    - kelas prediksi dan ground truth harus sama.

    Hasil ini untuk analisis pada satu threshold, bukan perhitungan AP.
    """
    gt = [
        box for box in ground_truth
        if box["class_id"] == class_id
    ]

    pred = [
        box for box in predictions
        if box["class_id"] == class_id
        and box["confidence"] >= CONF_THRESHOLD
    ]

    pred.sort(key=lambda item: item["confidence"], reverse=True)

    matched_gt = set()
    true_positive = 0

    for prediction in pred:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, target in enumerate(gt):
            if gt_index in matched_gt:
                continue

            iou = box_iou(prediction["xywh"], target["xywh"])

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= IOU_THRESHOLD:
            true_positive += 1
            matched_gt.add(best_gt_index)

    false_positive = len(pred) - true_positive
    false_negative = len(gt) - true_positive

    return true_positive, false_positive, false_negative


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


# 3. BACA METRIK SEBELUM DAN SESUDAH ADAPTATION

overall_rows = []
class_rows = []
all_metrics = {}

for model in MODEL_NAMES:
    for stage in ["before", "after"]:
        metrics_path = (
            RUN_DIR
            / "results_adapt"
            / model
            / stage
            / "metrics.json"
        )

        metrics = load_json(metrics_path)
        all_metrics[(model, stage)] = metrics

        overall_rows.append({
            "model": model,
            "kondisi": stage,
            "precision_%": metrics["precision"] * 100,
            "recall_%": metrics["recall"] * 100,
            "mAP50_%": metrics["map50"] * 100,
            "mAP50_95_%": metrics["map50_95"] * 100,
        })

        for class_name in ["smoke", "fire"]:
            class_metrics = metrics.get("per_class", {}).get(class_name)

            if class_metrics is None:
                print(
                    f"PERHATIAN: metrik kelas {class_name} "
                    f"tidak tersedia untuk {model} {stage}"
                )
                continue

            class_rows.append({
                "model": model,
                "kondisi": stage,
                "kelas": class_name,
                "precision_%": class_metrics["precision"] * 100,
                "recall_%": class_metrics["recall"] * 100,
                "AP50_%": class_metrics["ap50"] * 100,
                "AP50_95_%": class_metrics["ap50_95"] * 100,
            })

overall_df = pd.DataFrame(overall_rows)
class_df = pd.DataFrame(class_rows)

print("\n=== METRIK GABUNGAN TEST INDONESIA ===")
display(overall_df.round(2))

print("\n=== METRIK PER KELAS ===")
display(class_df.round(2))

overall_df.to_csv(OUTPUT_DIR / "metrik_gabungan.csv", index=False)
class_df.to_csv(OUTPUT_DIR / "metrik_per_kelas.csv", index=False)


# 4. HITUNG PERUBAHAN SEBELUM → SESUDAH

delta_rows = []

for model in MODEL_NAMES:
    before = all_metrics[(model, "before")]
    after = all_metrics[(model, "after")]

    for metric in ["precision", "recall", "map50", "map50_95"]:
        delta_rows.append({
            "model": model,
            "kelas": "gabungan",
            "metrik": metric,
            "sebelum_%": before[metric] * 100,
            "sesudah_%": after[metric] * 100,
            "delta_pp": (after[metric] - before[metric]) * 100,
        })

    for class_name in ["smoke", "fire"]:
        before_class = before.get("per_class", {}).get(class_name)
        after_class = after.get("per_class", {}).get(class_name)

        if not before_class or not after_class:
            continue

        for metric in ["precision", "recall", "ap50", "ap50_95"]:
            delta_rows.append({
                "model": model,
                "kelas": class_name,
                "metrik": metric,
                "sebelum_%": before_class[metric] * 100,
                "sesudah_%": after_class[metric] * 100,
                "delta_pp": (
                    after_class[metric] - before_class[metric]
                ) * 100,
            })

delta_df = pd.DataFrame(delta_rows)

print("\n=== PERUBAHAN SETELAH ADAPTATION ===")
display(delta_df.round(2))

delta_df.to_csv(OUTPUT_DIR / "perubahan_metrik.csv", index=False)


# 5. GRAFIK AP PER KELAS

if not class_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, class_name in zip(axes, ["smoke", "fire"]):
        subset = class_df[class_df["kelas"] == class_name]

        pivot = subset.pivot(
            index="model",
            columns="kondisi",
            values="AP50_%",
        ).reindex(MODEL_NAMES)

        pivot = pivot.reindex(columns=["before", "after"])

        pivot.plot.bar(
            ax=ax,
            rot=0,
            color=["#7b8da3", "#e56b2f"],
            width=0.72,
        )

        ax.set_title(f"AP@50 — {class_name}")
        ax.set_xlabel("")
        ax.set_ylabel("AP@50 (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(["Sebelum", "Sesudah"])

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "perbandingan_ap50_per_kelas.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.show()


# 6. KOMPOSISI TEST SET

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

test_images = sorted([
    path for path in TEST_IMAGES.iterdir()
    if path.suffix.lower() in image_extensions
])

assert test_images, "Tidak ada gambar pada folder test."

composition = Counter()
instance_counts = Counter()

for image_path in test_images:
    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Label test hilang: {label_path}")

    gt_boxes = load_yolo_boxes(label_path)
    classes_in_image = {box["class_id"] for box in gt_boxes}

    if classes_in_image == {0}:
        composition["smoke saja"] += 1
    elif classes_in_image == {1}:
        composition["fire saja"] += 1
    elif classes_in_image == {0, 1}:
        composition["smoke dan fire"] += 1
    elif not classes_in_image:
        composition["label kosong"] += 1
    else:
        composition["kelas lain"] += 1

    for box in gt_boxes:
        instance_counts[CLASS_NAMES.get(box["class_id"], "lainnya")] += 1

composition_df = pd.DataFrame([
    {"kategori": key, "jumlah_gambar": value}
    for key, value in composition.items()
])

instances_df = pd.DataFrame([
    {"kelas": key, "jumlah_bounding_box": value}
    for key, value in instance_counts.items()
])

print("\n=== KOMPOSISI TEST SET ===")
print("Jumlah gambar test:", len(test_images))
display(composition_df)

print("\n=== JUMLAH BOUNDING BOX TEST SET ===")
display(instances_df)

composition_df.to_csv(OUTPUT_DIR / "komposisi_test.csv", index=False)
instances_df.to_csv(OUTPUT_DIR / "jumlah_bbox_test.csv", index=False)


# 7. ANALISIS TP, FP, DAN FN PER GAMBAR DAN KELAS

detail_rows = []

for model in MODEL_NAMES:
    for stage in ["before", "after"]:
        prediction_dir = (
            RUN_DIR
            / "results_adapt"
            / model
            / stage
            / "plots"
            / "labels"
        )

        if not prediction_dir.is_dir():
            raise FileNotFoundError(
                f"Folder prediksi tidak ditemukan: {prediction_dir}\n"
                "Pastikan evaluasi notebook sebelumnya dijalankan dengan "
                "save_txt=True dan save_conf=True."
            )

        for image_path in test_images:
            ground_truth = load_yolo_boxes(
                TEST_LABELS / f"{image_path.stem}.txt"
            )

            predictions = load_yolo_boxes(
                prediction_dir / f"{image_path.stem}.txt",
                is_prediction=True,
            )

            for class_id, class_name in CLASS_NAMES.items():
                tp, fp, fn = match_detections(
                    ground_truth,
                    predictions,
                    class_id,
                )

                detail_rows.append({
                    "model": model,
                    "kondisi": stage,
                    "gambar": image_path.name,
                    "kelas": class_name,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "ground_truth": tp + fn,
                    "prediksi": tp + fp,
                })

detail_df = pd.DataFrame(detail_rows)

summary_df = (
    detail_df
    .groupby(["model", "kondisi", "kelas"], as_index=False)
    [["TP", "FP", "FN", "ground_truth", "prediksi"]]
    .sum()
)

summary_df["precision_threshold_%"] = summary_df.apply(
    lambda row: 100 * safe_divide(row["TP"], row["TP"] + row["FP"]),
    axis=1,
)

summary_df["recall_threshold_%"] = summary_df.apply(
    lambda row: 100 * safe_divide(row["TP"], row["TP"] + row["FN"]),
    axis=1,
)

print(
    "\n=== TP / FP / FN PADA "
    f"CONFIDENCE {CONF_THRESHOLD} DAN IoU {IOU_THRESHOLD} ==="
)
display(summary_df.round(2))

detail_df.to_csv(OUTPUT_DIR / "detail_kesalahan_per_gambar.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "ringkasan_tp_fp_fn.csv", index=False)


# 8. GAMBAR YANG PERLU DITINJAU

print("\n=== CONTOH GAMBAR DENGAN SMOKE TERLEWAT SETELAH ADAPTATION ===")

smoke_missed = (
    detail_df[
        (detail_df["kondisi"] == "after")
        & (detail_df["kelas"] == "smoke")
        & (detail_df["FN"] > 0)
    ]
    .sort_values(["FN", "FP"], ascending=False)
)

display(smoke_missed.head(20))

print("\n=== CONTOH GAMBAR DENGAN FALSE POSITIVE SETELAH ADAPTATION ===")

false_alarms = (
    detail_df[
        (detail_df["kondisi"] == "after")
        & (detail_df["FP"] > 0)
    ]
    .sort_values(["FP", "FN"], ascending=False)
)

display(false_alarms.head(20))

smoke_missed.to_csv(OUTPUT_DIR / "review_smoke_terlewat.csv", index=False)
false_alarms.to_csv(OUTPUT_DIR / "review_false_positive.csv", index=False)


# 9. TAMPILKAN CONFUSION MATRIX DAN PRECISION-RECALL CURVE

for model in MODEL_NAMES:
    for stage in ["before", "after"]:
        plots_dir = (
            RUN_DIR
            / "results_adapt"
            / model
            / stage
            / "plots"
        )

        print(f"\n=== {model.upper()} — {stage.upper()} ===")

        for filename in [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
        ]:
            image_path = plots_dir / filename

            if image_path.is_file():
                print(filename)
                display(IPImage(filename=str(image_path), width=650))
            else:
                print("Grafik tidak ditemukan:", image_path)


# 10. KOMPRES HASIL ANALISIS

zip_path = RUN_DIR / "analisis_lanjutan.zip"

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for file_path in OUTPUT_DIR.rglob("*"):
        if file_path.is_file():
            archive.write(
                file_path,
                arcname=file_path.relative_to(OUTPUT_DIR.parent),
            )

print("\nAnalisis selesai.")
print("Folder analisis:", OUTPUT_DIR)
print("ZIP analisis:", zip_path)

# %% [markdown]
# ## 9. Visualisasi kasus dan analisis ukuran objek
# Overlay ground truth (hijau) dan prediksi (merah) untuk ditinjau manual. Kelompok ukuran menggunakan luas bbox relatif terhadap gambar, bukan definisi COCO; hasil tiap kelompok bersifat deskriptif. File CSV memuat jumlah objek terlewat per ukuran dan sumber gambar bila tersedia.

# %%
from PIL import Image, ImageDraw
import math
preview_root = OUTPUT_DIR / 'preview_kesalahan'
preview_root.mkdir(parents=True, exist_ok=True)
manifest_rows = {r['image_filename']: r for r in csv.DictReader((DATASET_DIR / 'manifest.csv').open(encoding='utf-8-sig', newline=''))}
size_rows = []
for model in MODEL_NAMES:
    for stage in ('before', 'after'):
        pred_dir = RUN_DIR / 'results_adapt' / model / stage / 'plots' / 'labels'
        target_dir = preview_root / model / stage
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in test_images:
            gt = load_yolo_boxes(TEST_LABELS / f'{image_path.stem}.txt')
            pred = [b for b in load_yolo_boxes(pred_dir / f'{image_path.stem}.txt', True) if b['confidence'] >= CONF_THRESHOLD]
            im = Image.open(image_path).convert('RGB'); draw = ImageDraw.Draw(im); w, h = im.size
            def draw_box(b, color, prefix):
                x,y,bw,bh=b['xywh']; xy=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
                draw.rectangle(xy, outline=color, width=max(2, round(min(w,h)/250)))
                caption=prefix + ' ' + CLASS_NAMES[b['class_id']] + (f" {b['confidence']:.2f}" if b['confidence'] is not None else '')
                draw.text((max(0,xy[0]),max(0,xy[1]-12)), caption, fill=color, stroke_width=2, stroke_fill='black')
            for b in gt: draw_box(b, '#00e676', 'GT')
            for b in pred: draw_box(b, '#ff453a', 'PRED')
            im.save(target_dir / (image_path.stem + '.jpg'), quality=85)
            # Reuse same confidence-ordered one-to-one matching for GT-size attribution.
            for cls in CLASS_NAMES:
                group_gt=[b for b in gt if b['class_id']==cls]
                group_pred=sorted((b for b in pred if b['class_id']==cls), key=lambda b:b['confidence'], reverse=True)
                matched=set()
                for b in group_pred:
                    options=[(box_iou(b['xywh'], g['xywh']),i) for i,g in enumerate(group_gt) if i not in matched]
                    if options:
                        overlap,index=max(options)
                        if overlap >= IOU_THRESHOLD: matched.add(index)
                for i,g in enumerate(group_gt):
                    area=g['xywh'][2]*g['xywh'][3]
                    band='kecil <1%' if area < .01 else ('sedang 1–10%' if area < .10 else 'besar >=10%')
                    size_rows.append({'model':model,'stage':stage,'image':image_path.name,'class':CLASS_NAMES[cls], 'bbox_area_fraction':area,'size_group':band,'detected':i in matched,'video_id':manifest_rows.get(image_path.name,{}).get('video_id',''),'source_url':manifest_rows.get(image_path.name,{}).get('source_url','')})
size_df=pd.DataFrame(size_rows)
size_df.to_csv(OUTPUT_DIR/'analisis_per_objek_dan_ukuran.csv',index=False)
if not size_df.empty:
    size_summary=size_df.groupby(['model','stage','class','size_group'],dropna=False)['detected'].agg(['sum','count','mean']).reset_index()
    size_summary.columns=['model','stage','class','size_group','tp','ground_truth','recall_deskriptif']
    size_summary.to_csv(OUTPUT_DIR/'ringkasan_ukuran.csv',index=False)
    display(size_summary)
print('Preview (hijau GT; merah prediksi):',preview_root)

# %% [markdown]
# ## 10. Sensitivitas threshold pada validation
# Gunakan hasil ini untuk memahami trade-off; pemilihan threshold untuk klaim akhir perlu validation yang terpisah dan dibekukan sebelum menjalankan test independen. Tidak digunakan untuk mengganti hasil mAP. Gambar dengan label kosong harus diaudit sebagai negatif terlebih dahulu.

# %%
# Inference validation disimpan per run dengan confidence minimum 0.01.
# Perhitungan memakai matching sederhana yang sama, bukan evaluator AP resmi.
threshold_rows=[]
val_images=sorted(p for p in (PREPARED/'split/images/val').iterdir() if p.suffix.lower() in image_extensions)
for model_name in MODEL_NAMES:
    for stage, weights in [('before', WEIGHTS[model_name]), ('after', BEST[model_name])]:
        model=YOLO(str(weights)); check_model(model)
        val_predictions={}
        for im_path in val_images:
            result=model.predict(source=str(im_path), imgsz=IMGSZ, conf=0.01, device=DEVICE, verbose=False)[0]
            boxes=result.boxes
            val_predictions[im_path.name]=[{'class_id':int(c),'xywh':tuple(map(float,xywh)),'confidence':float(conf)} for c,xywh,conf in zip(boxes.cls.cpu().tolist(),boxes.xywhn.cpu().tolist(),boxes.conf.cpu().tolist())]
        for threshold in (0.10,0.25,0.50,0.75):
            CONF_THRESHOLD=threshold
            for cls, cls_name in CLASS_NAMES.items():
                tp=fp=fn=0
                for im_path in val_images:
                    gt=load_yolo_boxes(PREPARED/'split/labels/val'/f'{im_path.stem}.txt')
                    a,b,c=match_detections(gt,val_predictions[im_path.name],cls);tp+=a;fp+=b;fn+=c
                threshold_rows.append({'model':model_name,'stage':stage,'class':cls_name,'confidence':threshold,'iou':IOU_THRESHOLD,'TP':tp,'FP':fp,'FN':fn,'precision':safe_divide(tp,tp+fp),'recall':safe_divide(tp,tp+fn)})
CONF_THRESHOLD=ANALYSIS_CONF
threshold_df=pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_DIR/'threshold_validation_deskriptif.csv', index=False)
display(threshold_df)

# %% [markdown]
# ## 11. Ekspor hasil
# Arsip mencakup metrik, konfigurasi, split, checkpoint, grafik, dan analisis. Hasil tetap bersifat eksploratif. Dataset media dapat memiliki pembatasan izin; periksa sebelum membagikan arsip.

# %%
from IPython.display import FileLink
zip_path=shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name)
print('Arsip hasil:', zip_path)
display(FileLink(zip_path))
