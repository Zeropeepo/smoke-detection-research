# Domain shift deteksi asap dan api di Indonesia

Eksperimen ini membandingkan empat checkpoint sumber dan supervised fine-tuning pada data Indonesia. Run terdahulu bersifat eksploratif; pipeline matched tersedia untuk eksperimen lanjutan.

## Mulai di sini

- [`01_eksperimen_domain_shift.ipynb`](01_eksperimen_domain_shift.ipynb): persiapan split, pemanggilan runner, tiga evaluasi, dan ringkasan.
- [`kaggle/karhutla-yolo11n-yolo26n-adaptation-review.ipynb`](kaggle/karhutla-yolo11n-yolo26n-adaptation-review.ipynb): run matched empat model dan visual error analysis; memasang Ultralytics bila belum tersedia di image Kaggle.
- [`02_analisis_kesalahan.ipynb`](02_analisis_kesalahan.ipynb): visualisasi before/after empat model, kasus FN/FP, ukuran objek, dan indikator bbox yang terlewat.
- [`../scripts/Domain_Shift_Indonesia/run_experiments.py`](../scripts/Domain_Shift_Indonesia/run_experiments.py): runner Python sequential dengan pencatatan konfigurasi/hash, skip tahap yang selesai, dan resume dari `last.pt`.
- [`analisis_lanjutan/`](analisis_lanjutan/): helper dan unit test analisis bbox saja.
- [`arsip/hasil_yolo11n_26n_20261123/`](arsip/hasil_yolo11n_26n_20261123/): satu lokasi kanonis untuk artefak run terdahulu, termasuk split/label test, prediksi empat model, konfigurasi, dan CSV analisis. Gambar asli berada di luar repo.
- [`../scripts/notebooks/Domain_Shift_Indonesia/`](../scripts/notebooks/Domain_Shift_Indonesia/): mirror `.py` format percent dari notebook.

## Protokol matched

Fine-tuning target memakai `imgsz=640`, SGD, `lr0=0.001`, maksimum 50 epoch, patience 15, batch 16, seed 42, deterministic, dan kelas `0=smoke`, `1=fire`. Augmentasi acak dinonaktifkan sama untuk semua model: HSV, geometri, flip, mosaic, mixup, cutmix, copy-paste, BGR, dan multi-scale bernilai nol. Resize/letterbox tetap menjadi preprocessing.

Ini menyamakan **fine-tuning pada data Indonesia**. Bobot awal empat model tetap punya riwayat training source yang berbeda; hasil membandingkan transfer checkpoint yang tersedia, bukan eksperimen yang sepenuhnya mengisolasi arsitektur.

## Menjalankan di Kaggle

1. Notebook Kaggle menggunakan dataset Indonesia, empat checkpoint sumber, dan runner sebagai inputs privat.
2. Notebook memasang Ultralytics bila image Kaggle belum memilikinya, lalu mencatat versi paket dan GPU sebelum run.
3. Jalankan notebook eksperimen secara berurutan; runner menyimpan evaluasi, training, hash data/split, checkpoint, konfigurasi, dan ringkasan hasil.
4. Untuk analisis visual terpisah, set `KARHUTLA_RUN_DIR` ke folder hasil dan `KARHUTLA_IMAGES_DIR` ke `combined_manual_review`, lalu jalankan notebook 02.

Runner menggunakan GPU dan versi Ultralytics yang tersedia. Jika run terputus, jalankan ulang dengan folder output sama; tahap lengkap dilewati dan training parsial dilanjutkan dari `last.pt`. Untuk perubahan data/config, gunakan folder output baru. Jalankan model secara berurutan agar tidak berebut GPU.

## Cara membaca hasil dan batas klaim

Saat dijalankan, notebook 02 menampilkan delapan panel per gambar (empat model × sebelum/sesudah), termasuk lima kasus dengan FN/FP gabungan terbanyak setelah adaptation. Output lama tidak disimpan di Git. Indikator FN adalah heuristik, bukan penyebab kausal; periksa citra dan GT karena label hilang atau cakupan bbox berbeda dapat membuat deteksi benar terhitung sebagai FP.

Run arsip masih eksploratif dan tidak menggunakan konfigurasi augmentasi yang matched pada semua model. Test berisi 59 gambar yang sama, telah dilihat dalam analisis, dan belum independen per kejadian/lokasi. Gunakan test baru yang dikelompokkan per kejadian untuk klaim final. Untuk EWS video, ukur false alarm per jam dan jeda deteksi juga.

Gambar berasal dari pihak ketiga dan notebook menampilkan gambar inline. Periksa atribusi dan izin sebelum membagikan notebook atau artefak ke repo publik.
