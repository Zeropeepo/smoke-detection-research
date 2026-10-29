# Reproduksi paper Fire 2025

Implementasi [A Comparative Analysis of YOLOv9, YOLOv10, YOLOv11 for Smoke and Fire Detection](https://www.mdpi.com/2571-6255/8/1/26) untuk HPC. Notebook dan `fire2025_reproduction.py` harus berada pada folder yang sama. Tidak ada kode untuk memasang library. `YOLO_AUTOINSTALL=False` juga disetel sebelum Ultralytics diimpor; dataset dan bobot tidak diunduh otomatis.

## Yang disiapkan di HPC

Gunakan kernel Python yang **sudah** menyediakan PyTorch dengan CUDA, Ultralytics yang mendukung YOLOv9t/YOLOv10n/YOLO11, PyYAML, NumPy, Matplotlib dan Pillow. Jalankan sel pemeriksaan environment di notebook untuk melihat versi dan GPU yang terdeteksi. Driver yang dilaporkan `nvidia-smi` tidak menjamin bahwa PyTorch di kernel dapat memakai CUDA.

Unduh dan ekstrak dataset secara terpisah:

- [Smoke and Fire — arsip multipart dari dataset asal](https://github.com/vinchole/zzzccc/tree/main/dataset%20for%20papar). Pertahankan folder `train`, `valid`, dan `test`; setiap split berisi `images` dan `labels`. Gabungkan/ekstrak seluruh bagian arsip dengan perangkat yang tersedia di HPC. Arsip tersebut tidak memiliki `data.yaml`; periksa urutan kelas ID 0/1 dengan sumber dataset dan visualisasi notebook.
- [D-Fire — repo resmi dan tautan unduhan](https://github.com/gaia-solutions-on-demand/DFireDataset), termasuk [versi Kaggle siap YOLO](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo). Layout yang diharapkan: `data/{train,val,test}/{images,labels}` di bawah root D-Fire.

Sediakan bobot awal di satu folder. Berikut aset resmi yang dapat ditransfer ke HPC tanpa memasang library:

- [yolov9t.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt)
- [yolov10n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt)
- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [yolo11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

Bobot COCO ini merupakan asumsi inisialisasi; sumber dan versi bobot awal yang benar-benar digunakan penulis belum dipastikan. Versi file dan hash SHA-256 direkam. User bertanggung jawab memilih kernel dan alokasi GPU sesuai kebijakan HPC yang digunakan.

## Cara memakai

1. Buka `Fire2025_Reproduction.ipynb` dengan kernel HPC yang sudah siap.
2. Isi `SMOKE_ROOT`, `DFIRE_ROOT`, `WEIGHTS_DIR`, dan `OUTPUT_ROOT`. Atur `SMOKE_CLASSES` berdasarkan label sebenarnya. Gunakan `SHOW_SMOKE_LABEL_SAMPLES=True` untuk melihat bbox ID 0/1.
3. Mulai dengan `PREPARE_DATA=True` dan `RUN_TRAINING=False`. Audit semua label, periksa jumlah split dan contoh label bermasalah, serta tinjau daftar 13 job.
4. Atur `RUN_TRAINING=True`; notebook menjalankan semua job berurutan. Pilih `RUN_IDS` untuk menjalankan sebagian jika harus melanjutkan antrean dalam sesi terpisah.
5. Hasil per run tersimpan di `outputs/fire2025/training`, `evaluation`, dan `metadata`; `comparison.csv` menggabungkan referensi paper, evaluasi validation dan evaluasi test. Program menggunakan `best.pt` untuk kedua evaluasi.

Notebook tidak memulai training hanya karena dibuka. Run selesai dilewati ketika dijalankan ulang; run parsial dengan `last.pt` dicoba dilanjutkan. Folder parsial tanpa checkpoint dihentikan untuk ditinjau. Output dengan konfigurasi berbeda tidak akan ditimpa; pilih folder output baru untuk eksperimen baru.

## Cakupan dan asumsi

- Tabel 4: YOLOv9t, YOLOv10n, YOLO11n pada 50 dan 100 epoch, batch 16, Adam. Ukuran 640 dan pemakaian dataset Smoke and Fire adalah interpretasi dari konteks paper; Tabel 3 tidak menjabarkan semua parameter.
- Tabel 6: tiga model yang sama, 100 epoch, 640, batch 64, SGD dan parameter Tabel 5.
- Tabel 7 Smoke and Fire: YOLO11n/YOLO11x, 300 epoch, Tabel 5 dan 416 sesuai teks paper.
- Tabel 7 D-Fire: YOLO11n/YOLO11x, 100 epoch. Karena paper tidak merinci semua hyperparameter untuk kedua run ini, contoh kode memakai Tabel 5 dan 640 sebagai asumsi. `TABLE7_DFIRE_OVERRIDES` tersedia untuk menyelaraskan jika ada bukti tambahan.
- Seed 42, `deterministic=True`, `patience=0`, jumlah worker dan batch evaluasi adalah pilihan implementasi dan dicatat. Lihat `args.yaml` yang dihasilkan untuk seluruh default library yang aktif.
- D-Fire Kaggle yang diaudit memiliki train/val/test 14.122/3.099/4.306. Kode menggabungkan train+val, membuat 15.498/1.723 menggunakan seed tetap, dan mempertahankan test. Ini split rekonstruksi; kesamaan file test dan seed dengan paper belum diverifikasi.
- Dataset D-Fire yang diaudit memiliki 18 bbox luas nol serta 379 bbox melewati batas gambar, termasuk 8 bbox berdimensi >1. Dataset asli dipertahankan; perhatikan peringatan dan gambar yang dilewati oleh versi Ultralytics yang digunakan. Audit ini tidak menjamin seluruh gambar benar secara visual.
- Metrik pada paper tidak selalu menjelaskan split dan aturan pemilihan checkpoint secara lengkap. Tabel hasil memisahkan val dan test agar tidak terjadi klaim perbandingan setara secara otomatis.

## Hasil

Setiap run merekam konfigurasi dan versi environment, fingerprint bobot, checksum daftar split dan label (isi gambar belum di-hash), asumsi, checkpoint `best.pt`/`last.pt`, `results.csv`, plot Ultralytics, metrik bbox agregat, dan AP kelas smoke/fire. `comparison.csv` berisi seluruh 13 referensi paper; run yang belum selesai diberi status `not_finished`.

## Pemeriksaan kode tanpa training

Dari folder proyek, jalankan `python -m unittest discover -s tests -v`. Uji memakai data sintetis dan model tiruan, sehingga tidak memerlukan GPU atau dataset sebenarnya; ini bukan pengganti pengujian training di HPC.
