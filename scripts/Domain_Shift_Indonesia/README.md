# Runner domain shift Indonesia

`run_experiments.py` menjalankan tiap model secara sekuensial: evaluasi seluruh target (eksploratif), evaluasi test source-only, supervised fine-tuning pada train/validation Indonesia, lalu evaluasi pada test yang sama. `--stage` bisa menjalankan `evaluate-before`, `adapt`, atau `evaluate-after` terpisah. Default `all` menjalankan seluruhnya.

Sediakan folder hasil persiapan notebook dengan `full/data.yaml`, `split/data.yaml`, dan `split_manifest.csv`; siapkan checkpoint yang kelasnya `0=smoke`, `1=fire`. Contoh di Kaggle:

```bash
python run_experiments.py \
  --prepared /kaggle/working/karhutla_matched_aug_v1/prepared \
  --output /kaggle/working/karhutla_matched_aug_v1 \
  --weight yolov9t=/kaggle/input/checkpoints/yolov9t.pt \
  --weight yolov10n=/kaggle/input/checkpoints/yolov10n.pt \
  --weight yolo11n=/kaggle/input/checkpoints/yolo11n.pt \
  --weight yolo26n=/kaggle/input/checkpoints/yolo26n.pt
```

Default training: 640 px, 50 epoch, patience 15, batch 16, SGD, lr 0.001, seed 42. Semua augmentasi acak disetel nol untuk seluruh model. Edit konfigurasi hanya jika membuat folder output baru. Script merekam hash checkpoint, split, gambar dan label serta versi paket ke konfigurasi run; run yang tidak cocok tidak akan digabung. Jika ada `last.pt` dan config masih cocok, training dilanjutkan. Run yang punya `best.pt` dan ID konfigurasi sama dilewati.

Untuk notebook analisis, set `KARHUTLA_RUN_DIR` ke folder output dan `KARHUTLA_IMAGES_DIR` ke folder gambar. Output mencakup metrik, prediksi bbox Ultralytics, checkpoint, konfigurasi, split, label test, dan tabel ringkasan.
