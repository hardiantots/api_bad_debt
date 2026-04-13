# Bad Debt Early-Warning API

FastAPI service untuk scoring risiko bad debt berbasis snapshot feature engineering.

## Tujuan

- Menyediakan arsitektur hybrid: read dari MySQL, write hasil scoring ke SQLite lokal.
- Memisahkan proses compute (trigger async) dan proses baca hasil (read-only, paginated).
- Menjaga inference pipeline sinkron dengan artifacts model.
- Mendukung filtering customer affiliate Kalla Group dari hasil DB scoring.
- Siap dijalankan di VM Windows maupun Linux.

## Struktur Proyek

```text
Model API/
├── .env
├── api.py
├── README.md
├── requirements.txt
├── start_api.bat
├── start_api.sh
├── artifacts/
│   ├── stacked_recall_driven_model.joblib
│   ├── bad_debt_snapshot_lgbm_hyper_smote_16_features.joblib
│   ├── feature_cols_stacked.json
│   ├── feature_cols_snapshot_16_features.json
│   └── list_all_cust_affiliate_kalla.csv
└── bad_debt_app/
    ├── api/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── service.py
  │   ├── routes_compute.py
    │   ├── routes_db.py
    │   ├── routes_upload.py
    │   └── routes_system.py
    ├── data/
    │   ├── __init__.py
    │   ├── db.py
  │   ├── db_two_pass.py
  │   └── models.py
    └── feature_engineering/
        ├── __init__.py
        ├── io.py
        ├── base.py
        ├── pre_due.py
        ├── history.py
        ├── new_model.py
        └── pipeline.py
```

## Prasyarat

- Python 3.10+
- MySQL dengan tabel:
  - ar_invoice_list_2
  - ar_receipt_list
  - OracleCustomer

Catatan akses:

- Saat akses MySQL bersifat read-only, API tetap berjalan penuh karena hasil scoring disimpan di SQLite lokal.

## Instalasi

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Konfigurasi Environment

Buat file .env di root Model API:

```env
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=3306
DB_NAME=...

# Opsional keamanan API
API_KEY=

# Opsional runtime
API_HOST=0.0.0.0
API_PORT=8000
UVICORN_RELOAD=false

# Opsional threshold risiko
THRESHOLD_LOW=0.3
THRESHOLD_HIGH=0.6

# Opsional CORS, pisahkan dengan koma
CORS_ALLOW_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Scheduler compute otomatis
COMPUTE_SCHEDULE_HOUR=6
COMPUTE_AUTO_ENABLED=true
COMPUTE_DEFAULT_TIME_RANGE=3m
COMPUTE_KEEP_DAYS=30

# Auto publish score compute ke MySQL (invoice-level)
COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL=true
COMPUTE_PUBLISH_TARGET_TABLE=hasil_baddebt
```

Catatan:

- Endpoint /models tetap bisa merespons walau env DB belum lengkap, dengan fallback date range default.
- Dengan `COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL=true`, hasil compute invoice-level akan otomatis di-publish ke tabel MySQL terbaru (`COMPUTE_PUBLISH_TARGET_TABLE`) setelah compute sukses.
- Hasil `customer_risk` tetap disimpan di SQLite lokal.

## Menjalankan API

### Windows

```bat
start_api.bat
```

### Linux VM

```bash
chmod +x start_api.sh
./start_api.sh
```

### Linux VM + PM2 (Direkomendasikan untuk service)

File PM2 sudah disiapkan di `ecosystem.config.cjs` dengan nama app `api_bad_debt`.

1. Setup Python environment (sekali saja)

```bash
cd /path/ke/api_bad_debt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install PM2 tanpa sudo (via nvm)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install --lts
npm install -g pm2
pm2 -v
```

3. Start API dengan PM2

```bash
cd /path/ke/api_bad_debt
source .venv/bin/activate
pm2 start ecosystem.config.cjs --only api_bad_debt
pm2 status
pm2 logs api_bad_debt
```

4. Persist proses PM2

```bash
pm2 save
```

Catatan startup otomatis setelah reboot:

- `pm2 startup` biasanya mengeluarkan 1 command yang perlu dijalankan dengan sudo untuk registrasi systemd.
- Jika tidak punya akses sudo, alternatifnya jalankan `pm2 resurrect` via crontab user (`crontab -e`) dengan `@reboot`.

5. Update setelah git pull

```bash
cd /path/ke/api_bad_debt
git pull
source .venv/bin/activate
pip install -r requirements.txt
pm2 restart api_bad_debt --update-env
```

Atau langsung:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## Ringkasan Arsitektur Hybrid

1. Trigger compute via POST /db/compute.
2. API fetch data source dari MySQL (invoice, receipt, customer).
3. API jalankan feature engineering + scoring model.
4. Hasil invoice-level dan customer-level disimpan ke local_data/scoring.db.
5. Endpoint GET /db/\* hanya membaca hasil pre-computed dari SQLite lokal (read-only, paginated).
6. Jika dibutuhkan integrasi ke tabel MySQL terbaru, gunakan POST /db/compute/publish untuk push hasil compute ke tabel hasil_baddebt.

7. POST /db/compute/publish

- Fungsi: memindahkan hasil compute invoice-level dari SQLite ke tabel MySQL `hasil_baddebt`.
- Default: publish job completed terbaru berdasarkan model + snapshot + time_range.
- Bisa publish job spesifik dengan `job_id`.
- Mendukung `replace_partition=true` untuk menghapus partisi lama (source_model_key + source_snapshot_date + source_time_range) sebelum insert.

## Endpoint Utama

## Update Kontrak API (April 2026)

Bagian ini merangkum perubahan terbaru supaya frontend, QA, dan tim integrasi punya referensi cepat.

### A. Perubahan endpoint dan parameter

1. POST /db/compute

- Tetap sebagai trigger async compute.
- Response sukses: 202 Accepted + job_id.
- Jika job lain masih running: 409.

2. GET /db/score

- Tetap read-only dari SQLite (hasil pre-computed).
- Mendukung page, page_size, sort_by, sort_order, search, risk_level.
- Jika time_range=custom, start_date dan end_date wajib dikirim.
- Jika custom range tidak lengkap, response 422.

3. GET /db/customer_risk

- Mendukung page, page_size, sort_by, sort_order, search, risk_cust.
- Jika time_range=custom, start_date dan end_date wajib dikirim.
- Jika custom range tidak lengkap, response 422.

4. GET /db/alerts

- Mendukung threshold + pagination + sorting + search.
- Jika time_range=custom, start_date dan end_date wajib dikirim.

5. GET /db/early_warning/receipt_trigger

- Kini mendukung risk_level dan search (selain pagination/sorting).
- Jika time_range=custom, start_date dan end_date wajib dikirim.

6. GET /db/score_csv

- Tetap export seluruh hasil score pre-computed.
- Jika time_range=custom, start_date dan end_date wajib dikirim.

### B. Perubahan behavior penting

1. Pemilihan job untuk read-only endpoint lebih aman pada custom range.

- Lookup latest job tidak hanya model + snapshot_date + time_range.
- Untuk custom, start_date dan end_date ikut dipakai untuk menghindari mismatch antar window custom.

2. Scheduler auto-compute berjalan dengan snapshot harian dinamis.

- Snapshot date otomatis memakai tanggal hari ini saat scheduler men-trigger job.
- Jam tetap mengikuti COMPUTE_SCHEDULE_HOUR.

3. Default snapshot endpoint dihitung saat request.

- Jika parameter `snapshot_date` tidak dikirim, endpoint compute/read/upload akan memakai tanggal hari ini saat request dieksekusi.
- Ini mencegah default tanggal menjadi stale saat service berjalan lama.

4. Validasi no pre-compute tetap konsisten.

- Jika belum ada hasil compute untuk kombinasi parameter yang diminta, endpoint read-only mengembalikan 404 + hint untuk menjalankan POST /db/compute.

5. Penyimpanan hasil compute memakai strategi replace partition.

- Untuk kombinasi partisi yang sama (model_key + snapshot_date + time_range, dan khusus custom juga start_date + end_date), data hasil compute lama akan dihapus dulu sebelum insert hasil baru.
- Tujuan: mencegah duplikasi row lintas job dan memastikan perubahan invoice terbaru menimpa hasil lama pada partisi yang sama.

6. Sequencing replace partition dibuat aman untuk read path.

- Job baru menyimpan hasil dulu, lalu job ditandai completed, setelah itu barulah data partisi lama dibersihkan.
- Dengan urutan ini, endpoint GET tetap bisa membaca data completed sebelumnya selama compute baru masih running (menghindari gap data sementara).

### C. Status code yang perlu di-handle client

1. 200: Read endpoint sukses.
2. 202: Compute job berhasil ditrigger.
3. 404: Hasil pre-compute belum tersedia untuk parameter tersebut.
4. 409: Compute ditolak karena masih ada job running.
5. 422: Parameter custom range tidak lengkap (start_date/end_date wajib).

### 1) System Endpoint

- GET /health
  - Fungsi: cek status service, default model, snapshot default, threshold, dan daftar model aktif.
  - Kapan dipakai: monitoring uptime, smoke test deploy, validasi startup.

- GET /models
  - Fungsi: metadata model dan opsi time range untuk frontend.
  - Output utama:
    - models: daftar model pada registry.
    - time_ranges: 1w, 2w, 1m, 3m, 6m, 1y, all, custom.
    - min_date, max_date: rentang data dari DB (fallback jika DB belum siap).

### 2) Compute Endpoint (Write ke SQLite)

- POST /db/compute
  - Fungsi: trigger scoring background (async).
  - Jika ada job running lain, API merespons 409.
  - Output: 202 Accepted + job_id.

- GET /db/compute/status
  - Fungsi: status job terakhir.

- GET /db/compute/status/{job_id}
  - Fungsi: status detail job tertentu.

- GET /db/compute/history
  - Fungsi: list histori compute jobs.

### 3) DB Read-only Endpoint (Baca Hasil Pre-computed)

Semua endpoint ini tidak menjalankan scoring real-time. Data dibaca dari SQLite berdasarkan hasil compute terbaru.

Parameter umum (sesuai endpoint):

- model: stacked atau lgbm_hyper_smote.
- snapshot_date: tanggal acuan scoring, format YYYY-MM-DD.
- time_range: 1w, 2w, 1m, 3m, 6m, 1y, all, custom.
- page, page_size: pagination.
- sort_by, sort_order: sorting.
- search, risk_level/risk_cust: filtering.

- GET /db/score
  - Fungsi: list hasil score invoice (paginated).
  - Output utama:
    - last_computed_at, job_id, total_invoices.
    - pagination.
    - risk_summary, high_risk_count.
    - preview: daftar skor invoice.
    - customer_risk_summary.
    - top_efl_invoices (ranking expected financial loss).

- GET /db/customer_risk
  - Fungsi: list agregasi customer risk (paginated).
  - Output utama: pagination, customer_risk_summary, customer_risk.

- GET /db/alerts
  - Parameter tambahan: threshold (0.0-1.0).
  - Fungsi: hanya menampilkan invoice dengan prob_bad_debt >= threshold (paginated).
  - Output utama: alerts_count, pagination, alerts, risk_summary.

- GET /db/score_csv
  - Fungsi: export seluruh hasil score pre-computed dalam format CSV.
  - Output: file CSV attachment.

- GET /db/early_warning/receipt_trigger
  - Fungsi: mode early warning dari data pre-computed (paginated).
  - Output utama: processed_invoices, alerts_count, high_risk_count, all_scores_preview, top_efl_invoices.

### 4) Upload-based Scoring Endpoint

Semua endpoint upload menerima multipart/form-data:

- invoice_csv (required)
- receipt_csv (required)
- customer_json (optional)
- model (optional, default stacked)
- snapshot_date (optional, default dari config)
- customer_format (optional, misalnya csv/json/parquet)

Daftar endpoint:

- POST /score: output JSON (preview hasil scoring).
- POST /score_csv: output CSV attachment.
- POST /alerts: output JSON terfilter threshold.
- POST /early_warning/receipt_trigger: output JSON mode early warning.

Catatan:

- Pada flow upload, jika customer_json tidak dikirim maka proses tetap berjalan tanpa fallback OracleCustomer_slim.

## Alur Kerja API

### A. Flow Compute (Hybrid)

1. Client memanggil POST /db/compute.
2. API membuat compute job status=running.
3. Background task menjalankan fetch raw data dari MySQL.
4. Pipeline feature engineering + model scoring dijalankan.
5. Hasil invoice score disimpan ke bad_debt_score_results (SQLite).
6. Hasil customer risk disimpan ke bad_debt_customer_risk (SQLite).
7. Job diupdate menjadi completed + metadata summary.
8. Replace partition: data hasil job lama pada partisi yang sama dibersihkan (untuk mencegah duplikasi).
9. Data job lama (retention) dibersihkan sesuai COMPUTE_KEEP_DAYS.

Catatan partisi replace:

- Kunci partisi standard: model_key + snapshot_date + time_range.
- Kunci partisi custom: model_key + snapshot_date + time_range + start_date + end_date.
- Histori compute job tetap disimpan untuk audit, tetapi row hasil data lama pada partisi yang sama akan digantikan.

### B. Flow Read-only DB Endpoint

1. Client memanggil GET /db/score atau endpoint DB read lain.
2. API cari latest completed job untuk kombinasi model + snapshot_date + time_range.
3. Jika time_range=custom, start_date dan end_date ikut dipakai untuk memilih job yang tepat.
4. API query data dari SQLite menggunakan pagination/filter/sort.
5. API kirim response terstruktur dengan pagination metadata.

### C. Flow Scheduler

1. Saat startup, API memastikan tabel SQLite tersedia.
2. Jika COMPUTE_AUTO_ENABLED=true, APScheduler aktif.
3. Scheduler menjalankan auto compute setiap hari jam COMPUTE_SCHEDULE_HOUR.
4. Snapshot date untuk auto compute diisi otomatis dengan tanggal berjalan (dinamis harian).

### D. Flow Upload

1. Client upload invoice_csv dan receipt_csv (customer_json opsional).
2. Service membaca file, validasi ukuran upload, lalu parse menjadi dataframe.
3. Pipeline feature engineering dan scoring model berjalan seperti flow DB.
4. Response dikirim sesuai endpoint (JSON/CSV).

### E. Flow Security Middleware

1. Jika API_KEY kosong, request diproses normal.
2. Jika API_KEY terisi, endpoint non-public harus kirim header:
   - X-API-Key: <key>
   - atau Authorization: Bearer <key>
3. Jika key tidak valid, API mengembalikan 401.

## Detail Flow Filter Affiliate

Untuk proses compute dari data DB:

- Setelah data customer periode scoring didapat dan proses scoring selesai, sistem menghapus customer yang termasuk daftar affiliate di artifacts/list_all_cust_affiliate_kalla.csv.
- Penghapusan diterapkan sebelum data disimpan ke SQLite sehingga endpoint GET /db/\* otomatis membaca data yang sudah bersih.

## Keamanan

Jika API_KEY di-set, endpoint selain health/docs membutuhkan header:

- X-API-Key: <key>
- atau Authorization: Bearer <key>

## Catatan Deployment VM

- Set UVICORN_RELOAD=false untuk production.
- Jalankan di balik reverse proxy (Nginx/Apache) jika endpoint diekspos publik.
- Simpan .env sebagai secret file VM, jangan commit ke git.
- Pastikan folder local_data writable oleh proses API (untuk file scoring.db).
- Gunakan process manager:
  - Linux: systemd/supervisor
  - Windows Server: NSSM atau Task Scheduler service mode

## Reinstall Bersih di VM (Hapus Progress Lama)

Gunakan langkah ini jika ingin reset deployment lama lalu install ulang dari repo `api_bad_debt`.

1. Stop proses lama

```bash
pm2 stop api_bad_debt || true
pm2 delete api_bad_debt || true
pm2 save || true
pkill -f "uvicorn api:app" || true
```

2. Backup file penting lama (opsional)

```bash
mkdir -p "$HOME/backup_bad_debt"
cp -f /path/ke/api_bad_debt/.env "$HOME/backup_bad_debt/.env.$(date +%F_%H%M%S)" || true
cp -f /path/ke/api_bad_debt/local_data/scoring.db "$HOME/backup_bad_debt/scoring.$(date +%F_%H%M%S).db" || true
```

3. Hapus folder deployment lama

```bash
rm -rf /path/ke/api_bad_debt
```

4. Clone ulang + install dependency

```bash
cd /path/ke
git clone <url-repo-anda> api_bad_debt
cd api_bad_debt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

5. Restore/isi `.env`

```bash
cp -f "$HOME/backup_bad_debt/.env.*" .env 2>/dev/null || true
# Jika file tidak ada, buat .env manual sesuai template di atas.
```

6. Jalankan ulang dengan PM2

```bash
pm2 start ecosystem.config.cjs --only api_bad_debt
pm2 save
pm2 status
pm2 logs api_bad_debt
```

## Contoh Quick Start Hybrid

1. Trigger compute:

```bash
curl -X POST "http://localhost:8000/db/compute?model=stacked&snapshot_date=2026-03-15&time_range=1w"
```

2. Cek status job:

```bash
curl "http://localhost:8000/db/compute/status"
```

3. Baca hasil paginated:

```bash
curl "http://localhost:8000/db/score?model=stacked&snapshot_date=2026-03-15&time_range=1w&page=1&page_size=50&sort_by=prob_bad_debt&sort_order=desc"
```

## Health Check

```bash
curl http://localhost:8000/health
```
