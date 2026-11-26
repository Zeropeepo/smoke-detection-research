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
# # Analisis kesalahan deteksi asap dan api pada data Indonesia
#
# Empat checkpoint sumber diuji sebelum dan sesudah supervised fine-tuning pada **test yang sama (59 gambar)**. Analisis ini menjelaskan objek yang cocok, terlewat, dan prediksi yang tidak cocok; hasil masih eksploratif karena anotasi dan provenance belum selesai diaudit.
#
# Panel output yang saat ini tersimpan berasal dari run arsip dengan augmentasi berbeda antar-model, jadi catatan visualnya bersifat eksploratif. Setelah matched run selesai, jalankan ulang notebook ini agar panel dan catatan kuantitatif diperbarui. Untuk perbandingan setara, jalankan ulang empat model dengan `scripts/Domain_Shift_Indonesia/run_experiments.py`: split target, parameter training, dan kebijakan tanpa augmentasi acak disamakan. Pilih folder hasil baru melalui variabel lingkungan `KARHUTLA_RUN_DIR`. Test yang ditinjau di sini bukan lagi test buta; gunakan kejadian independen untuk klaim akhir.
#
# Sumber: `KARHUTLA_RUN_DIR` dari runner yang sama atau arsip analisis lama; gambar dan anotasi: dataset `combined_manual_review`. Gambar asli tidak disalin ke repo. Tentukan `KARHUTLA_IMAGES_DIR` bila dataset tidak berada pada path default. Gambar visual dapat mengandung materi pihak ketiga; periksa hak pakai sebelum distribusi publik.

# %%
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import display

ROOT = Path(os.environ.get('KARHUTLA_REPO', str(Path.cwd()))).expanduser()
if not (ROOT / 'analisis_lanjutan' / 'bbox_analysis.py').exists():
    ROOT = ROOT / 'Domain_Shift_Indonesia'
run_value = os.environ.get('KARHUTLA_RUN_DIR')
if not run_value:
    kaggle_run = Path('/kaggle/working/karhutla_matched_aug_v1')
    run_value = str(kaggle_run) if kaggle_run.is_dir() else None
RUN_OUTPUT = Path(run_value).expanduser() if run_value else None
if RUN_OUTPUT and not (RUN_OUTPUT.is_dir() and (RUN_OUTPUT / 'results_adapt').is_dir()):
    raise FileNotFoundError(f'KARHUTLA_RUN_DIR bukan output run yang lengkap: {RUN_OUTPUT}')
if RUN_OUTPUT:
    ART = RUN_OUTPUT
    PRED_ROOT = ART / 'results_adapt'
    GT_ROOT = ART / 'ground_truth'
else:
    ART = ROOT / 'arsip' / 'hasil_yolo11n_26n_20261123'
    PRED_ROOT = ART / 'results_adapt'
    GT_ROOT = ART / 'prepared' / 'split' / 'labels' / 'test'
sys.path.insert(0, str((ROOT / 'analisis_lanjutan').resolve()))
from bbox_analysis import CLASSES, MODELS, read_boxes, box_iou, audit_case, dataset_audit
IMAGE_DIR = Path(os.environ.get('KARHUTLA_IMAGES_DIR', str((ROOT / '../../dataset/combined_manual_review').resolve())))
CONF, IOU = 0.25, 0.50
print('Data prediksi:', PRED_ROOT.resolve())
print('Gambar:', IMAGE_DIR.resolve(), 'tersedia:', IMAGE_DIR.is_dir())

# %% [markdown]
# ## 1. Data dan protokol
#
# Notebook membaca satu split test, satu kumpulan label, dan file prediksi dari satu run. Run baru menyimpan hash file gambar/anotasi, split, checkpoint, versi Ultralytics, serta konfigurasi training di `run_config.json`. P/R, AP, dan mAP berasal dari evaluator Ultralytics; hitungan TP/FP/FN di gambar memakai confidence 0,25 dan IoU 0,50 sebagai alat inspeksi.
#
# Jika memilih hasil arsip lama, augmentasi antar-model berbeda dan hasilnya harus dibaca sebagai eksplorasi. Untuk hasil setara, arahkan `KARHUTLA_RUN_DIR` ke output `run_experiments.py`.

# %%
manifest_path = ART / 'split_manifest.csv'
if not manifest_path.is_file():
    manifest_path = ART / 'prepared' / 'split_manifest.csv'
manifest = pd.read_csv(manifest_path)
test = manifest[manifest['split'] == 'test'].copy()
assert len(test) == 59 and len(list(GT_ROOT.glob('*.txt'))) == 59
print('Gambar test:', len(test), '| hash gambar unik:', test['sha256_image'].nunique())
if IMAGE_DIR.is_dir():
    mismatches = []
    for row in test.itertuples(index=False):
        path = IMAGE_DIR / row.image_filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256_image:
            mismatches.append(row.image_filename)
    if mismatches:
        raise ValueError(f'Gambar hilang atau hash berubah: {mismatches[:10]}')
    print('Hash 59 gambar cocok dengan split manifest.')
display(test.loc[test.duplicated('sha256_image', keep=False), ['image_filename','sha256_image','group']].sort_values('sha256_image'))

if (ART / 'results_adapt').is_dir():
    configs=[]
    for model in MODELS:
        path=ART/'results_adapt'/model/'run_config.json'
        if path.is_file():
            cfg=json.loads(path.read_text())
            configs.append({'model':model,'run_id':cfg['run_id'],**cfg['config']})
    common=['imgsz','epochs','patience','batch','optimizer','lr0','seed','workers','augmentation']
    mismatches = {}
    for key in common:
        values={json.dumps(row[key],sort_keys=True) for row in configs}
        if len(values)>1:
            mismatches[key] = sorted(values)
    if mismatches:
        print('PERINGATAN: konfigurasi arsip berbeda antar-model; hasil ini hanya eksploratif:', mismatches)
    elif RUN_OUTPUT:
        print('Konfigurasi matched terverifikasi pada folder output ini.')
    display(pd.DataFrame([{k:row[k] for k in ['model','run_id','imgsz','epochs','patience','batch','optimizer','lr0','seed','workers']} for row in configs]))
    if configs:
        print('Augmentasi target yang dipakai semua model:')
        display(pd.Series(configs[0]['augmentation'],name='value').to_frame())

# %% [markdown]
# ## 2. Perubahan performa pada test yang sama
#
# Delta ditulis dalam poin persentase. Panel gambar di bawah memperlihatkan apa yang berada di balik metrik: objek yang terdeteksi, bbox yang meleset, serta false alarm yang perlu ditinjau terhadap anotasi. mAP bukan akurasi per gambar.

# %%
metric_rows=[]
for model in MODELS:
    for stage in ('before','after'):
        d=json.loads((PRED_ROOT/model/stage/'metrics.json').read_text())
        metric_rows.append({'model':model,'stage':stage,'mAP50':100*d['map50'],'mAP50-95':100*d['map50_95'],
                            'AP50 smoke':100*d['per_class']['smoke']['ap50'],'AP50 fire':100*d['per_class']['fire']['ap50']})
metrics=pd.DataFrame(metric_rows)
display(metrics.round(2))
fig,ax=plt.subplots(figsize=(8,4))
pivot=metrics.pivot(index='model',columns='stage',values='mAP50').reindex(MODELS)
pivot[['before','after']].plot.bar(ax=ax,rot=0,color=['#8795a1','#bf5b27'])
ax.set(title='mAP50 pada test Indonesia sementara',xlabel='',ylabel='mAP50 (%)',ylim=(0,100))
ax.legend(['Sebelum','Sesudah']);plt.tight_layout();plt.show()

# %% [markdown]
# ## 3. FP/FN pada threshold tetap
#
# Keempat model dihitung dengan confidence 0,25 dan IoU matching 0,50. Ini pencocokan greedy sederhana satu GT per prediksi, bukan AP resmi. Kelas salah, bbox tidak cukup overlap, anotasi hilang, serta deteksi duplikat dapat muncul sebagai FP/FN. Satu gambar dapat mengandung beberapa bbox; angka tidak berarti jumlah kejadian kebakaran.

# %%
rows=[]
for model in MODELS:
    for stage in ('before','after'):
        rows.extend(dataset_audit(ART,model,stage,confidence=CONF,threshold=IOU))
errors=pd.DataFrame(rows)
summary=errors.groupby(['model','stage','class'],as_index=False)[['tp','fp','fn']].sum()
summary['precision']=summary['tp']/(summary['tp']+summary['fp']).replace(0,np.nan)
summary['recall']=summary['tp']/(summary['tp']+summary['fn']).replace(0,np.nan)
display(summary.round(3))
assert len(errors)==4*2*59*2

# %% [markdown]
# ## 4. Perbandingan visual pada gambar yang sama
#
# Setiap baris membandingkan sebelum dan sesudah adaptation untuk satu model; keempat model melihat gambar identik. Hijau = ground truth, oranye = prediksi smoke, magenta = prediksi fire. TP/FP/FN dihitung dengan confidence ≥0,25 dan IoU ≥0,50. Kotak yang tampak salah bisa berasal dari anotasi yang hilang atau berbeda cakupan, jadi telaah gambarnya sebelum menilai model.

# %% [markdown]
# **Asal gambar dan izin:** `ours_108.jpg` tercatat sebagai ANTARA Foto/Iggoy el Fitra, [sumber](https://www.antarafoto.com/id/view/2577749/karhutla-di-solok); izin redistribusi perlu diperoleh. `ours_045.jpg` tercatat di [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kebakaran_Hutan_Ambon.jpg) sebagai PUSDATIN Kemendagri; metadata lisensi perlu dicocokkan sebelum publikasi. `dito_001.jpg` identik secara hash dengan `ours_045.jpg`, tetapi paket Dito tidak menyertakan sumbernya. Provenance `dito_078.jpeg` dan `dito_132.jpeg` belum tersedia. Output notebook memuat gambar; jangan menganggap semua bebas didistribusikan.

# %%
def image_for(stem):
    rows = test[test.image_filename.map(lambda x: Path(x).stem) == stem]
    if len(rows)!=1:
        raise ValueError(f'Nama test tidak unik: {stem}')
    path=IMAGE_DIR / rows.iloc[0].image_filename
    if not path.is_file():
        raise FileNotFoundError(f'Gambar tidak tersedia: {path}. Atur KARHUTLA_IMAGES_DIR.')
    return path

def paint(image, boxes, color, is_prediction=False):
    draw=ImageDraw.Draw(image); w,h=image.size
    thickness=max(2,round(min(w,h)/260))
    for b in boxes:
        if is_prediction and b['confidence'] < CONF: continue
        x,y,bw,bh=b['xywh']; xy=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
        outline=('#d65f00' if b['class_id']==0 else '#c31c9c') if is_prediction else color
        draw.rectangle(xy,outline=outline,width=thickness)
        label=('P ' if is_prediction else 'GT ')+CLASSES[b['class_id']]
        if is_prediction: label+=f" {b['confidence']:.2f}"
        draw.text((max(0,xy[0]),max(0,xy[1]-12)),label,fill=outline,stroke_width=2,stroke_fill='black')
    return image

def show_case(stem):
    path=image_for(stem)
    gt=read_boxes(GT_ROOT/f'{stem}.txt')
    base=Image.open(path).convert('RGB')
    fig,axes=plt.subplots(4,2,figsize=(13,17))
    for i,model in enumerate(MODELS):
        for j,stage in enumerate(('before','after')):
            pred=read_boxes(PRED_ROOT/model/stage/'plots'/'labels'/f'{stem}.txt',prediction=True)
            frame=paint(base.copy(),gt,'#00ee73')
            frame=paint(frame,pred,None,True)
            result=errors[(errors.image_stem==stem)&(errors.model==model)&(errors.stage==stage)][['tp','fp','fn']].sum()
            axes[i,j].imshow(frame);axes[i,j].axis('off')
            axes[i,j].set_title(f"{model} • {stage} | TP {result.tp} / FP {result.fp} / FN {result.fn}", fontsize=10)
    fig.suptitle(path.name+' | GT hijau, prediksi smoke oranye, fire magenta',fontsize=13)
    plt.tight_layout();plt.show()

if IMAGE_DIR.is_dir(): show_case('dito_078')
else: print('Preview dilewati; atur KARHUTLA_IMAGES_DIR ke folder gambar asli.')

# %% [markdown]
# ### Contoh: api terhalang vegetasi dan anotasi duplikat — `dito_001.jpg` dan `ours_045.jpg`
#
# Dua file ini memiliki hash gambar identik, tetapi label berbeda. `dito_001` memakai beberapa bbox api; `ours_045` memakai satu bbox api lebih luas. Jika prediksi berfokus pada bagian nyala api sedangkan GT mencakup area luas, IoU bisa rendah. Ini bukti ketidakkonsistenan anotasi; prediksi tidak otomatis benar. Hitungan dua file bukan dua kejadian independen.

# %%
if IMAGE_DIR.is_dir():
    show_case('dito_001')
    show_case('ours_045')

# %% [markdown]
# ### Contoh: asap menyebar — `ours_108.jpg`
#
# Batas plume transparan dan bertumpuk sulit ditetapkan. Untuk YOLO11n after, salah satu prediksi memiliki IoU sekitar 0,498 terhadap GT terdekat, sehingga gagal pada ambang 0,50 meskipun kotaknya tumpang tindih. Peninjauan bersama diperlukan untuk memisahkan kesalahan lokalisasi dari ambiguitas anotasi.

# %%
if IMAGE_DIR.is_dir(): show_case('ours_108')

# %% [markdown]
# ### Contoh: titik api pada malam hari — `dito_132.jpeg`
#
# Pencahayaan rendah dan beberapa objek kecil terlihat pada gambar. Ini contoh untuk review, bukan bukti bahwa model secara umum gagal pada semua kondisi malam. Untuk menguji pengaruh malam, beri label kondisi pada seluruh test dan periksa jumlah kejadian per kelompok.

# %%
if IMAGE_DIR.is_dir(): show_case('dito_132')

# %% [markdown]
# ## 5. Gambar dengan error terbanyak setelah adaptation
#
# Daftar diurutkan menurut jumlah FN lalu FP yang terakumulasi pada empat model. Gambar dengan SHA-256 sama hanya dimasukkan sekali agar duplikat tidak mendominasi. Setiap gambar ditampilkan dalam 8 panel (empat model × sebelum/sesudah); ranking memilih kasus untuk ditinjau dan tidak membuktikan penyebab error.

# %%
hard = errors[errors.stage == 'after'].groupby('image_stem', as_index=False)[['fn','fp']].sum()
image_keys = test.assign(image_stem=test.image_filename.map(lambda name: Path(name).stem))
hard = (hard.merge(image_keys[['image_stem','image_filename','sha256_image']], on='image_stem', how='left')
        .sort_values(['fn','fp'], ascending=False)
        .drop_duplicates('sha256_image')
        .head(5))
display(hard[['image_filename','fn','fp']])
if IMAGE_DIR.is_dir():
    for row in hard.itertuples(index=False):
        print(f'Kasus prioritas: {row.image_stem} | FN={row.fn}, FP={row.fp}')
        show_case(row.image_stem)
else:
    print('Gambar belum tersedia. Atur KARHUTLA_IMAGES_DIR ke folder combined_manual_review.')

# %% [markdown]
# ## 6. Kapan bbox fire terlewat?
#
# Ukuran dihitung sebagai luas bbox GT dibagi luas gambar: kecil <1%, sedang 1–10%, besar ≥10%. Ini bukan definisi COCO dan bukan jarak fisik. Ukuran objek saja tidak membuktikan penyebab; bbox dan kejadian saling berkorelasi.

# %%
size_rows=[]
for label in sorted(GT_ROOT.glob('*.txt')):
    gt=read_boxes(label)
    for model in MODELS:
        for stage in ('before','after'):
            pred=read_boxes(PRED_ROOT/model/stage/'plots'/'labels'/label.name,prediction=True)
            for class_id,name in CLASSES.items():
                result=audit_case(gt,pred,class_id,CONF,IOU)
                matched={i for i,_,_ in result['matches']}
                for i,box in enumerate(gt):
                    if box['class_id'] != class_id: continue
                    area=box['xywh'][2]*box['xywh'][3]
                    size='kecil <1%' if area<.01 else ('sedang 1–10%' if area<.1 else 'besar ≥10%')
                    size_rows.append({'model':model,'stage':stage,'kelas':name,'ukuran':size,'detected':int(i in matched),'image':label.stem})
sizes=pd.DataFrame(size_rows)
size_summary=sizes.groupby(['model','stage','kelas','ukuran'],as_index=False).agg(terdeteksi=('detected','sum'),total=('detected','size'))
size_summary['recall']=size_summary.terdeteksi/size_summary.total
display(size_summary[(size_summary.stage=='after') & (size_summary.kelas=='fire')].round(3))

# %% [markdown]
# ## 7. Indikator untuk setiap bbox yang terlewat
#
# Kategori berikut dihitung pada prediksi yang telah tersimpan setelah post-processing: kandidat kelas sama di bawah confidence, overlap parsial (IoU 0,10–<0,50), kemungkinan prediksi kelas lain, kandidat yang sudah dipasangkan, atau tidak ada kandidat terdekat. Urutan kategori tercatat dalam fungsi `fn_indicators`. Kategori adalah **indikator heuristik, bukan diagnosis sebab**. Kandidat yang sama dapat menjadi kandidat untuk beberapa GT; karena itu jumlah kandidat di bawah threshold bukan jumlah yang pasti pulih bila threshold diturunkan.
#
# Tabel kasus memuat semua empat model, sebelum dan sesudah. False positive dan false negative juga dapat disebabkan anotasi tidak lengkap atau beda granularitas. Confidence 0,25 dan IoU 0,50 bukan threshold yang telah dioptimalkan untuk EWS.

# %%
from bbox_analysis import fn_indicators
indicators=[]
for model in MODELS:
    for stage in ('before','after'):
        for label in sorted(GT_ROOT.glob('*.txt')):
            gt=read_boxes(label)
            pred=read_boxes(PRED_ROOT/model/stage/'plots'/'labels'/label.name,prediction=True)
            for class_id,name in CLASSES.items():
                for item in fn_indicators(gt,pred,class_id,CONF,IOU):
                    indicators.append({'model':model,'stage':stage,'kelas':name,'gambar':label.stem,**item})
indicators=pd.DataFrame(indicators)
indicator_counts=indicators.groupby(['model','stage','kelas','category'],as_index=False).size().rename(columns={'size':'bbox'})
display(indicator_counts[(indicator_counts.stage=='after') & (indicator_counts.kelas=='fire')])
assert len(indicators)==int(summary['fn'].sum())

# %%
for stem in ['dito_078','dito_001','ours_045','ours_108','dito_132']:
    print('Kasus:',stem)
    case=errors[(errors.image_stem==stem)].copy()
    display(case.pivot_table(index=['model','stage'],columns='class',values=['tp','fp','fn']).fillna(0).astype(int))

# %% [markdown]
# ## 8. Catatan audit visual manual pada panel run arsip
#
# Aku memeriksa panel yang sudah tertanam untuk `dito_078.jpeg`, `dito_001.jpg`, `ours_108.jpg`, dan `dito_132.jpeg`. Semua angka berikut berasal dari run lama yang augmentasinya tidak sama antar-model. Gunakan sebagai contoh jenis error yang perlu dicari lagi setelah matched run, bukan sebagai peringkat model.
#
# - **`dito_078.jpeg`, asap jauh di atas lereng dan beberapa titik api kecil.** Bbox smoke mencakup area plume yang luas; bbox fire kecil dan terpisah. Di panel after, keempat model masih melewatkan 9–11 objek menurut matching threshold tetap. Prediksi smoke beberapa kali sangat lebar sampai mencakup sebagian langit/lereng, sedangkan beberapa api kecil terlihat namun bbox-nya tidak selalu cocok dengan anotasi. Ini mendukung audit khusus untuk objek kecil/jauh dan konsistensi aturan apakah satu bbox fire meliputi satu titik atau satu gugus.
# - **`dito_001.jpg`, asap tebal di hutan dengan beberapa lokasi api.** Model dapat mengenali sebagian flame, tetapi prediksi fire sering bertumpuk/terduplikasi di lokasi sama; matching one-to-one menghitung prediksi tambahan sebagai FP. GT smoke juga luas dan bertumpang tindih dengan area api, sehingga batas kelas dan kebijakan pemisahan fire/smoke perlu diperiksa sebelum menyalahkan model.
# - **`ours_108.jpg`, asap tipis menyebar pada gambar udara dengan watermark.** Semua panel after menampilkan dua FN pada ambang IoU 0,50, meskipun kotak prediksi tampak menutup sebagian plume. Batas asap memang samar dan cakupan bbox prediksi berbeda dari GT; sebagian kasus seperti ini bisa gagal IoU karena perbedaan luasan kotak, bukan karena model sama sekali tidak melihat asap. Sumber udara dan watermark juga membuat gambar ini kurang mewakili CCTV menara.
# - **`dito_132.jpeg`, garis api malam hari dengan asap bercahaya.** Panel lama menunjukkan hitungan FN after lebih kecil pada YOLOv9t, YOLOv10n, dan YOLO11n, tetapi tidak pada YOLO26n. Satu frame malam tidak cukup untuk menyimpulkan kemampuan malam; periksa beberapa video/kejadian dan negatif malam untuk membandingkan alarm palsu.
#
# Pola yang paling jelas dari audit visual ini adalah **sulitnya menentukan batas plume asap dan konsistensi granularitas bbox api**, ditambah objek kecil/jauh. Ini hipotesis yang perlu diuji pada hasil matched dan label yang ditinjau manusia. False positive yang bertumpuk atau tampak masuk akal bisa pula menunjukkan duplikasi prediksi atau GT yang belum lengkap.

# %% [markdown]
# ## 9. Cara membaca pola error dan batas kesimpulan
#
# - Cocokkan pola pada beberapa gambar: objek sangat kecil/jauh, asap transparan, malam, vegetasi menutup objek, api dan asap tumpang tindih, atau haze yang menyerupai smoke. Contoh tunggal hanya menunjukkan kasus, bukan membuktikan penyebab umum.
# - Periksa false positive pada citra asli. Jika objek nyata tidak dianotasi, hitungan FP bisa mengukur kekurangan label, bukan kesalahan model.
# - Periksa false negative terhadap kebijakan anotasi: batas plume samar dan kumpulan api dapat diberi kotak dengan cakupan berbeda. Model bisa gagal IoU meski lokalisasi terlihat dekat.
# - Untuk klaim domain adaptation, gunakan event/lokasi terpisah saat membentuk test. Test yang sudah dibuka di analisis ini hanya memberi bukti eksploratif.
# - Model EWS perlu diukur pada rangkaian video, termasuk alarm palsu per jam dan jeda deteksi; metrik bbox per frame belum cukup.
#
# ### Literatur pendukung
#
# - Llerena dkk., *Noise-Aware Evaluation of Object Detectors*, WACV 2025: [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf).
# - Schubert dkk., *Identifying Label Errors in Object Detection Datasets by Loss Inspection*, WACV 2024: [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Schubert_Identifying_Label_Errors_in_Object_Detection_Datasets_by_Loss_Inspection_WACV_2024_paper.pdf).
# - *Early Wildfire Smoke Detection with a Multi-Resolution Framework and Two-Stage Classification Pipeline*, Fire 2026: [paper](https://www.mdpi.com/2571-6255/9/2/92). Literatur ini mendukung tantangan umum deteksi asap awal; ia tidak membuktikan penyebab error pada sampel Indonesia kita.
