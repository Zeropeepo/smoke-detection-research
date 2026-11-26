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
# # Fire 2025 — eksperimen 5–7 berurutan
# Menjalankan YOLO11n 50 epoch, YOLO11n 100 epoch, dan YOLOv9t 100 epoch SGD. Setiap run dievaluasi pada val dan test. Data sementara disimpan di /tmp, hasil di /kaggle/working.
#

# %%
import os, sys, subprocess, importlib.metadata, pathlib, urllib.request
from pathlib import Path
os.environ['YOLO_AUTOINSTALL'] = 'False'
print('Python:', sys.version, flush=True)
try:
    print('ultralytics:', importlib.metadata.version('ultralytics'))
except importlib.metadata.PackageNotFoundError:
    # Instalasi hanya di sesi Kaggle, tidak mengubah HPC.
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], check=True)
    print('ultralytics:', importlib.metadata.version('ultralytics'))
import torch
print('CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'GPU Kaggle tidak terdeteksi'
print('GPU:', torch.cuda.get_device_name(0))


# %%
# Ambil kode yang sama dengan versi repo yang sudah dipush.
work = Path('/tmp/fire2025-inputs')
work.mkdir(parents=True, exist_ok=True)
repo = 'https://raw.githubusercontent.com/Zeropeepo/smoke-detection-research/6cb776f/Fire2025_Reproduction/'
for name in ('fire2025_reproduction.py', 'run_hpc.py', 'extract_split_zip.py'):
    urllib.request.urlretrieve(repo + name, work / name)
    print('Kode tersedia:', name, flush=True)


# %%
# Unduh arsip Smoke and Fire yang digunakan paper, lalu periksa dan ekstrak.
parts = work / 'smoke_download'
parts.mkdir(exist_ok=True)
base = 'https://raw.githubusercontent.com/vinchole/zzzccc/main/dataset%20for%20papar/dataset1.'
for suffix in ['z01','z02','z03','z04','z05','z06','z07','z08','z09','zip']:
    target = parts / ('dataset1.' + suffix)
    print('Mengunduh', target.name, flush=True)
    urllib.request.urlretrieve(base + suffix, target)
subprocess.run([sys.executable, str(work / 'extract_split_zip.py'),
                str(parts / 'dataset1.zip'), str(work / 'Smoke_and_Fire')], check=True)


# %%
# Cari root D-Fire pada dataset input Kaggle, tanpa menebak lokasi mount.
matches = []
for images in Path('/kaggle/input').rglob('images'):
    if images.parent.name == 'train' and images.parent.parent.name == 'data':
        root = images.parent.parent.parent
        if all((root / 'data' / split / 'images').is_dir() for split in ('train', 'val', 'test')):
            matches.append(root)
if len(matches) != 1:
    raise RuntimeError(f'Root D-Fire tidak unik atau tidak ditemukan: {matches}')
dfire = matches[0]
print('D-Fire:', dfire, flush=True)
weights = work / 'weights'
weights.mkdir(exist_ok=True)
for model in ('yolo11n', 'yolov9t'):
    urllib.request.urlretrieve(f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{model}.pt', weights / f'{model}.pt')


# %%
# Training dan evaluasi tunggal. Dataset audit dan versi environment dicatat runner.
run = [sys.executable, '-u', str(work / 'run_hpc.py'), '--mode', 'train',
       '--smoke-root', str(work / 'Smoke_and_Fire'), '--dfire-root', str(dfire),
       '--weights-dir', str(weights), '--output-root', '/kaggle/working/fire2025-results',
       '--smoke-classes', 'smoke', 'fire', '--run-ids', 't4_yolo11n_50', 't4_yolo11n_100', 't6_yolov9t_100']
print('Memulai:', ' '.join(run), flush=True)
subprocess.run(run, check=True)
