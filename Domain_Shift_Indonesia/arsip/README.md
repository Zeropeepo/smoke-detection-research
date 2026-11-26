# Arsip eksperimen domain shift

`hasil_yolo11n_26n_20261123/` menyimpan satu snapshot analisis eksploratif. Empat model berbagi 59 gambar test dan label yang sama; hasil ini **tidak matched** karena konfigurasi training/augmentasi sumber berbeda. Gambar mentah tidak disalin ke repo.

- `prepared/`: laporan split, manifest, dan label test kanonis.
- `results_adapt/<model>/{before,after}/`: metrik evaluator dan prediksi YOLO per gambar untuk empat model.
- `results_adapt/<model>/training/args.yaml`: konfigurasi training lama yang tersedia.
- `analysis/`: CSV ringkasan kesalahan, ukuran bbox, perubahan metrik, dan indikator bbox.
- `ringkasan_tiga_evaluasi.csv`: ringkasan metrik run yang sebelumnya tercatat.
- `provenance.json`: pemeriksaan hash sumber, test, dan anotasi.

Run baru yang matched ditulis ke folder output Kaggle/HPC tersendiri dan tidak dicampur ke snapshot ini.
