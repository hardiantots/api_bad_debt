# Bad Debt Early-Warning API

FastAPI service untuk scoring risiko bad debt berbasis snapshot feature engineering.

## Tujuan

- Menyediakan endpoint scoring dari database dan upload file.
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
    │   ├── routes_db.py
    │   ├── routes_upload.py
    │   └── routes_system.py
    ├── data/
    │   ├── __init__.py
    │   ├── db.py
    │   └── db_two_pass.py
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
```

Catatan:

- Endpoint models tetap bisa merespons walau env DB belum lengkap, dengan fallback date range default.

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

Atau langsung:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## Endpoint Utama

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

### 2) DB-backed Scoring Endpoint

Semua endpoint DB mendukung parameter umum berikut (sesuai endpoint):

- model: stacked atau lgbm_hyper_smote.
- snapshot_date: tanggal acuan scoring, format YYYY-MM-DD.
- time_range: 1w, 2w, 1m, 3m, 6m, 1y, all, custom.
- start_date, end_date: dipakai jika time_range=custom.

- GET /db/score
  - Fungsi: scoring invoice hasil fetch DB.
  - Output utama:
    - total_invoices, risk_summary, high_risk_count.
    - preview: daftar skor invoice.
    - customer_risk_summary, customer_risk.
    - top_efl_invoices (ranking expected financial loss).

- GET /db/alerts
  - Parameter tambahan: threshold (0.0-1.0).
  - Fungsi: hanya menampilkan invoice dengan prob_bad_debt >= threshold.
  - Output utama: alerts_count, alerts, risk_summary + customer_risk.

- GET /db/score_csv
  - Fungsi: hasil score DB dalam format CSV download.
  - Output: file CSV attachment.

- GET /db/early_warning/receipt_trigger
  - Fungsi: mode early warning (analisis pre-due).
  - Output utama: processed_invoices, alerts_count, high_risk_count, all_scores_preview, customer_risk.

### 3) Upload-based Scoring Endpoint

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

### A. Flow DB (disarankan untuk produksi)

1. Client memanggil endpoint DB dengan model, snapshot_date, dan filter periode.
2. Service fetch data invoice, receipt, customer dari database (strategi two-pass untuk efisiensi).
3. Pipeline feature engineering membentuk fitur snapshot/pre-due.
4. Service memuat model artifact + schema fitur, lalu menghitung probabilitas bad debt.
5. Sistem membentuk output invoice-level (risk_level, recommended_action, expected_financial_loss).
6. Sistem membentuk agregasi customer_risk.
7. Sistem menjalankan filter affiliate dari artifacts/list_all_cust_affiliate_kalla.csv.
8. Response dikirim ke client (JSON/CSV tergantung endpoint).

### B. Flow Upload

1. Client upload invoice_csv dan receipt_csv (customer_json opsional).
2. Service membaca file, validasi ukuran upload, lalu parse menjadi dataframe.
3. Pipeline feature engineering dan scoring model berjalan seperti flow DB.
4. Response dikirim sesuai endpoint (JSON/CSV).

### C. Flow Security Middleware

1. Jika API_KEY kosong, request diproses normal.
2. Jika API_KEY terisi, endpoint non-public harus kirim header:
   - X-API-Key: <key>
   - atau Authorization: Bearer <key>
3. Jika key tidak valid, API mengembalikan 401.

## Detail Flow Filter Affiliate

Untuk endpoint DB:

- Setelah data customer periode mingguan didapat dan proses scoring selesai, sistem menghapus customer yang termasuk daftar affiliate di artifacts/list_all_cust_affiliate_kalla.csv.
- Penghapusan dilakukan pada hasil invoice list (preview, alerts, csv output) dan juga pada customer_risk.

## Keamanan

Jika API_KEY di-set, endpoint selain health/docs membutuhkan header:

- X-API-Key: <key>
- atau Authorization: Bearer <key>

## Catatan Deployment VM

- Set UVICORN_RELOAD=false untuk production.
- Jalankan di balik reverse proxy (Nginx/Apache) jika endpoint diekspos publik.
- Simpan .env sebagai secret file VM, jangan commit ke git.
- Gunakan process manager:
  - Linux: systemd/supervisor
  - Windows Server: NSSM atau Task Scheduler service mode

## Health Check

```bash
curl http://localhost:8000/health
```
