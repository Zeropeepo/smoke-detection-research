# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reproduksi Fire 2025 — deteksi asap dan api
#
# Paper: [A Comparative Analysis of YOLOv9, YOLOv10, YOLOv11 for Smoke and Fire Detection](https://www.mdpi.com/2571-6255/8/1/26). Notebook ini menyiapkan 13 konfigurasi eksperimen penulis, memeriksa dataset, melatih model secara berurutan, serta membandingkan metrik. Lihat `README.md` untuk penempatan data dan batasan reproduksi.
#
# **Tidak ada perintah instalasi library.** Tidak ada unduhan otomatis dataset atau bobot. Secara default training dan penyiapan data dinonaktifkan; ubah sakelar di bagian konfigurasi saat siap menjalankan di HPC.

# %% [markdown]
# ## 1. Impor kode dan periksa environment
#
# Simpan `fire2025_reproduction.py` dalam folder yang sama dengan notebook. Library yang dibutuhkan harus tersedia pada kernel HPC yang sudah dipilih.

# %%
import os, sys, json
from pathlib import Path
from pprint import pprint

# Tetapkan sebelum mengimpor Ultralytics, agar library tidak memasang dependensi otomatis.
os.environ["YOLO_AUTOINSTALL"] = "False"
PROJECT_DIR = Path.cwd().resolve()
if not (PROJECT_DIR / "fire2025_reproduction.py").is_file():
    possible = PROJECT_DIR / "Fire2025_Reproduction"
    if (possible / "fire2025_reproduction.py").is_file():
        PROJECT_DIR = possible
    else:
        raise FileNotFoundError("Buka notebook dari folder yang juga berisi fire2025_reproduction.py")
sys.path.insert(0, str(PROJECT_DIR))
import fire2025_reproduction as repro
print("Project:", PROJECT_DIR)
pprint(repro.environment_info(device=0))

# %% [markdown]
# ## 2. Sesuaikan path dan pilihan run
#
# Ubah tiga path sesuai lokasi **hasil ekstraksi** dan bobot `.pt` di HPC. Dataset Smoke and Fire yang diperiksa tidak memiliki `data.yaml`; tentukan urutan label ID 0 dan 1 melalui visualisasi pada langkah 3 atau sumber anotasi aslinya. D-Fire versi Kaggle yang diperiksa memakai ID 0 = smoke dan 1 = fire.
#
# `DEVICE=0` mengacu ke GPU pertama yang **terlihat oleh proses** setelah pengaturan scheduler/CUDA_VISIBLE_DEVICES. GPU 80 GB yang dipakai bersama proses lain dapat memiliki memori bebas lebih kecil; ukuran batch tidak diubah otomatis.

# %%
# WAJIB: sesuaikan dengan path pada filesystem HPC.
SMOKE_ROOT = Path("/path/ke/Smoke_and_Fire")
DFIRE_ROOT = Path("/path/ke/DFire")
WEIGHTS_DIR = Path("/path/ke/weights")
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "fire2025"

# Contoh setelah terverifikasi: ["fire", "smoke"] atau ["smoke", "fire"].
# Jangan menebak urutan kelas dataset pertama.
SMOKE_CLASSES = None
DFIRE_CLASSES = ["smoke", "fire"]

DEVICE = 0
WORKERS = 4
SEED = 42                    # Asumsi, bukan seed yang diverifikasi dari paper.
EVAL_BATCH = 16              # Asumsi; evaluasi val dan test dicatat terpisah.
DFIRE_VAL_COUNT = 1723

SHOW_SMOKE_LABEL_SAMPLES = False
PREPARE_DATA = False         # True untuk audit dan membuat manifest.
RUN_TRAINING = False         # True untuk melatih eksperimen terpilih.
EXPORT_COMPARISON = False    # True untuk membuat comparison.csv dari hasil yang ada.
SHOW_EXAMPLE_PREDICTIONS = False
PREDICT_SOURCE = Path("/path/ke/contoh_gambar_atau_video")

# Kosong berarti semua 13 run. Isi ID jika perlu melanjutkan subset tertentu.
RUN_IDS = []

# Tabel 7 D-Fire: hyperparameter tidak lengkap; default program adalah asumsi Tabel 5
# (640, batch 64, SGD). Contoh override: {"batch": 32}; catat sebagai deviasi.
TABLE7_DFIRE_OVERRIDES = {}

# %% [markdown]
# ## 3. Periksa urutan kelas Smoke and Fire
#
# Jika belum tahu arti ID 0 dan 1, atur `SHOW_SMOKE_LABEL_SAMPLES = True`. Bbox dari setiap ID ditampilkan pada contoh gambar; konfirmasi urutan dengan sumber anotasi, lalu isi `SMOKE_CLASSES`. Sel ini membutuhkan Matplotlib dan Pillow yang sudah ada di environment.

# %%
if SHOW_SMOKE_LABEL_SAMPLES:
    repro.show_label_samples(SMOKE_ROOT, samples_per_class=4, seed=SEED)
else:
    print("Visualisasi label dilewati. Pastikan SMOKE_CLASSES sebelum membuat manifest.")

# %% [markdown]
# ## 4. Tinjau matriks eksperimen
#
# Tabel 4 berisi dua jadwal terpisah (50 dan 100 epoch) untuk tiga model. Tabel 6 mengubah hyperparameter. Untuk run 300 epoch, paper menyatakan konfigurasi Tabel 5 dengan resolusi **416**. Paper belum menjelaskan seluruh detail run D-Fire; asumsi tersebut terlihat pada setiap job dan metadata.

# %%
ALL_JOBS = repro.build_experiments(TABLE7_DFIRE_OVERRIDES)
JOBS = [j for j in ALL_JOBS if not RUN_IDS or j["id"] in RUN_IDS]
unknown = set(RUN_IDS) - {j["id"] for j in ALL_JOBS}
if unknown:
    raise ValueError(f"ID run tidak dikenal: {sorted(unknown)}")
print(f"Eksperimen terpilih: {len(JOBS)} dari {len(ALL_JOBS)}")
for j in JOBS:
    a = j["train_args"]
    print(f"{j['id']:28} {j['dataset']:10} {j['model']:9} "
          f"epoch={j['epochs']:3} size={a['imgsz']:3} batch={a['batch']:2} {a['optimizer']}")

# %% [markdown]
# ## 5. Audit data dan buat split
#
# Setelah path dan kelas benar, atur `PREPARE_DATA = True`. Audit membaca label tanpa mengubahnya. Manifest D-Fire menggabungkan train dan val dari unduhan, lalu membuat 1.723 validation dengan seed tetap dan mempertahankan test unduhan. Ini **rekonstruksi**, karena seed dan identitas split penulis belum terverifikasi. Manifest dan hash-nya disimpan ke `OUTPUT_ROOT/data_manifests`.
#
# Layout default:
#
# ```text
# Smoke_and_Fire/{train,valid,test}/{images,labels}/
# DFire/data/{train,val,test}/{images,labels}/
# ```
#
# Versi D-Fire yang diaudit berisi beberapa bbox luas nol atau melewati batas gambar. Audit mencatat jumlahnya; data asli tidak dibersihkan secara diam-diam.

# %%
provenance = None
if PREPARE_DATA:
    provenance = repro.prepare_datasets(
        SMOKE_ROOT, DFIRE_ROOT, OUTPUT_ROOT, SMOKE_CLASSES, DFIRE_CLASSES,
        dfire_val_count=DFIRE_VAL_COUNT, split_seed=SEED,
    )
    pprint(provenance["datasets"])
    print("\nRingkasan audit label:")
    for split, result in provenance["audit_original_splits"].items():
        print(split, {k: v for k, v in result.items() if k != "examples"})
else:
    print("Persiapan data dilewati. Ubah PREPARE_DATA menjadi True setelah konfigurasi siap.")

# %% [markdown]
# ## 6. Jalankan antrean training
#
# Letakkan empat bobot awal dalam `WEIGHTS_DIR`: `yolov9t.pt`, `yolov10n.pt`, `yolo11n.pt`, dan `yolo11x.pt`. Bobot tersebut diasumsikan pretrained COCO; bobot persis yang digunakan penulis belum terverifikasi. File dan hash bobot direkam.
#
# Jika training terputus, jalankan ulang notebook dengan konfigurasi yang sama. Run lengkap dilewati dan run parsial dilanjutkan dari `last.pt` bila tersedia. Jika terjadi OOM atau dependensi tidak kompatibel, training berhenti dengan pesan kesalahan; konfigurasi tidak diubah otomatis.

# %%
if RUN_TRAINING:
    if provenance is None:
        raise RuntimeError("Jalankan persiapan data pada langkah 5 dengan PREPARE_DATA=True terlebih dahulu.")
    pprint(repro.require_training_environment(DEVICE))
    pprint(repro.validate_weight_files(WEIGHTS_DIR, JOBS))
    results = repro.run_queue(JOBS, provenance, WEIGHTS_DIR, OUTPUT_ROOT,
                              device=DEVICE, workers=WORKERS, seed=SEED,
                              eval_batch=EVAL_BATCH, continue_incomplete=True)
    print("Run selesai:", len(results))
else:
    print("Training tidak dijalankan. Atur RUN_TRAINING=True untuk memulai antrean.")

# %% [markdown]
# ## 7. Ringkasan dan perbandingan
#
# Evaluasi `best.pt` disimpan terpisah untuk validation dan test, beserta AP per kelas smoke/fire. `comparison.csv` menyimpan angka paper berdampingan dengan hasil keduanya; jangan menyatakan selisih sebagai perbandingan setara jika split evaluasi atau pengaturan paper belum diketahui.

# %%
if RUN_TRAINING or EXPORT_COMPARISON:
    comparison_path, rows = repro.summarize_results(ALL_JOBS, OUTPUT_ROOT)
    print("Tabel:", comparison_path)
    for row in rows:
        print(row["run_id"], row["status"],
              "paper mAP50=", row["paper_map50"],
              "val mAP50=", row.get("val_map50"),
              "test mAP50=", row.get("test_map50"),
              "test smoke AP50=", row.get("test_smoke_ap50"))
else:
    print("Perbandingan akan dibuat setelah training atau dengan EXPORT_COMPARISON=True.")

# %% [markdown]
# ## 8. Prediksi gambar atau video
#
# Opsional setelah training. Sesuaikan checkpoint dengan hasil yang ingin ditunjukkan pada PPT. Hasil inferensi disimpan dalam folder output, dan model menggunakan label dataset yang sesuai dengan run tersebut.

# %%
if SHOW_EXAMPLE_PREDICTIONS:
    YOLO = repro.load_ultralytics_without_install().YOLO
    CHECKPOINT = OUTPUT_ROOT / "training" / "t6_yolo11n_100" / "weights" / "best.pt"
    if not CHECKPOINT.is_file() or not PREDICT_SOURCE.exists():
        raise FileNotFoundError("Periksa lokasi checkpoint dan PREDICT_SOURCE terlebih dahulu.")
    prediction_model = YOLO(str(CHECKPOINT))
    prediction_model.predict(source=str(PREDICT_SOURCE), device="", save=True,
                             project=str(OUTPUT_ROOT / "predictions"), name="example", exist_ok=False)
else:
    print("Prediksi contoh dilewati.")

# %% [markdown]
# ## 9. Interpretasi untuk laporan
#
# Laporkan versi paket/kernel, GPU yang terlihat, konfigurasi aktual dari `args.yaml`, seed, hash bobot, split dan audit label. Jelaskan bahwa beberapa pengaturan, checkpoint awal, serta split D-Fire penulis tidak tersedia lengkap. Dokumentasikan jumlah anotasi yang diperingatkan/diabaikan framework dan evaluasi pada dua kelas serta AP kelas smoke.
