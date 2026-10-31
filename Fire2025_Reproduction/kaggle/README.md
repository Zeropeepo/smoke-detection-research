# Notebook eksperimen Kaggle

Salinan sumber notebook yang digunakan untuk eksperimen berikut:

- `fire2025-yolov10n-100.ipynb`: run 4, `t4_yolov10n_100`.
- `fire2025-runs-5-7.ipynb`: run 5–7 secara berurutan.
- `fire2025-run8-yolov10n-sgd.ipynb`: run 8, `t6_yolov10n_100`.

Notebook mengunduh kode runner pada commit `6cb776f`, dataset Smoke and Fire, dan bobot awal. Tambahkan dataset Kaggle `sayedgamal99/smoke-fire-detection-yolo` sebagai input dan aktifkan GPU serta internet. Notebook memasang Ultralytics di sesi Kaggle jika belum tersedia; ini terpisah dari environment HPC. Run 4 dan 5–7 tidak mengunci versi instalasi, sedangkan run 8 menggunakan `8.4.192` jika paket belum tersedia. Hasil yang direproduksi dapat berbeda bila versi environment berubah.

Notebook ini tidak menyertakan hasil, data, bobot, atau kredensial. Status dan metrik run harus diperiksa pada output masing-masing eksperimen.
