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

# %%
import os, sys, subprocess, importlib.metadata, urllib.request, json, hashlib
from pathlib import Path
os.environ['YOLO_AUTOINSTALL'] = 'False'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.4.192'], check=True)
import torch, ultralytics
from ultralytics import YOLO
print('Ultralytics:', ultralytics.__version__, 'Torch:', torch.__version__, flush=True)
assert torch.cuda.is_available(), 'GPU tidak tersedia'
print('GPU:', torch.cuda.get_device_name(0), flush=True)


# %%
work = Path('/tmp/fire2025-yolo26n'); work.mkdir(exist_ok=True)
parts = work / 'smoke_download'; parts.mkdir(exist_ok=True)
base = 'https://raw.githubusercontent.com/vinchole/zzzccc/main/dataset%20for%20papar/dataset1.'
for suffix in ['z01','z02','z03','z04','z05','z06','z07','z08','z09','zip']:
    print('Mengunduh', suffix, flush=True)
    urllib.request.urlretrieve(base + suffix, parts / ('dataset1.' + suffix))
extract = work / 'extract_split_zip.py'
urllib.request.urlretrieve('https://raw.githubusercontent.com/Zeropeepo/smoke-detection-research/6cb776f/Fire2025_Reproduction/extract_split_zip.py', extract)
root = work / 'Smoke_and_Fire'
subprocess.run([sys.executable, str(extract), str(parts / 'dataset1.zip'), str(root)], check=True)
for split in ['train', 'valid', 'test']:
    images = list((root / split / 'images').glob('*'))
    print(split, len(images), flush=True)
    assert len(images) == {'train':8494, 'valid':2114, 'test':1059}[split]


# %%
output = Path('/kaggle/working/yolo26n-smoke-fire'); output.mkdir(parents=True, exist_ok=True)
data = {'path': str(root), 'train': 'train/images', 'val': 'valid/images', 'test': 'test/images', 'names': {0: 'smoke', 1: 'fire'}}
data_file = output / 'data.yaml'
data_file.write_text(json.dumps(data, indent=2))
args = dict(data=str(data_file), epochs=100, imgsz=640, batch=64, optimizer='SGD', lr0=0.01, warmup_epochs=3.0, warmup_bias_lr=0.1, momentum=0.8, weight_decay=0.0005, amp=True, seed=42, deterministic=True, patience=0, device=0, workers=2, project=str(output), name='training', exist_ok=False, plots=True, save=True)
(output / 'experiment.json').write_text(json.dumps({'model':'yolo26n.pt','training_args':args,'ultralytics':ultralytics.__version__,'torch':torch.__version__,'dataset':'Smoke and Fire','note':'Baseline baru, bukan eksperimen paper; versi library dan workers dapat berbeda dari run sebelumnya.'}, indent=2))
model = YOLO('yolo26n.pt')
model.train(**args)


# %%
best = output / 'training' / 'weights' / 'best.pt'
assert best.is_file(), 'Checkpoint best.pt tidak ditemukan'
print('Evaluasi checkpoint:', best, flush=True)
for split in ('val','test'):
    m = YOLO(str(best)).val(data=str(data_file), split=split, imgsz=640, batch=8, device=0, workers=2, project=str(output / 'evaluation'), name=split, exist_ok=True, plots=True)
    b = m.box
    metrics = {'split':split, 'checkpoint':'best.pt', 'precision':float(b.mp), 'recall':float(b.mr), 'map50':float(b.map50), 'map50_95':float(b.map), 'per_class':{str(m.names[int(c)]):{'ap50':float(b.ap50[i]),'ap50_95':float(b.ap[i]),'precision':float(b.p[i]),'recall':float(b.r[i])} for i,c in enumerate(b.ap_class_index)}}
    dest = output / 'evaluation' / split / 'metrics.json'
    dest.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
print('SELESAI', flush=True)
