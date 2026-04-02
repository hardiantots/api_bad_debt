# Bad Debt Early-Warning API (16-Feature Pipeline)

Sistem deteksi dini risiko _bad debt_ menggunakan **FastAPI**, didukung oleh **Stacked Model (16 fitur)** dan integrasi langsung ke database MySQL.

## 🚀 Fitur Utama

- **16-Feature ML Pipeline**: Menghitung 9 fitur historis dan 4 fitur _survival_ secara dinamis untuk akurasi prediksi tinggi.
- **Dynamic Time Filtering**: Mendukung _Quick Select_ (1w, 1m, dll.) dan _Custom Date Range_ (Dari - Ke) yang dibatasi secara otomatis oleh ketersediaan data di database.
- **Flexible Data Source**: Dapat mengambil data langsung dari database atau melalui _upload_ file CSV.
- **Auto-Anchoring**: Perhitungan rentang waktu secara otomatis merujuk pada tanggal transaksi terbaru di database, bukan waktu server saat ini.

## 🔄 Sinkronisasi Dua Flow Notebook

API wrapper ini sudah dipetakan ke dua flow training notebook terbaru:

- `stacked`:
  - Artifact model: `artifacts/stacked_recall_driven_model.joblib`
  - Schema fitur: `artifacts/feature_cols_stacked.json`
- `lgbm_hyper_smote`:
  - Artifact model: `artifacts/bad_debt_snapshot_lgbm_hyper_smote_16_features.joblib`
  - Schema fitur: `artifacts/feature_cols_snapshot_16_features.json`

Keduanya memakai flow data terbaru yang sama pada inference:

- Credit memo netting berbasis `PREVIOUS_CUSTOMER_TRX_ID` sampai `snapshot_date`.
- Feature engineering pre-due + historikal + survival (16 fitur).
- Konvensi label training: `y_bad_debt_ever` (3 kondisi) dari notebook.

Saat memanggil endpoint, pilih model lewat parameter `model`.
Contoh:

```bash
# DB scoring dengan flow stacked
curl "http://localhost:8000/db/score?model=stacked&snapshot_date=2026-02-06&time_range=1m"

# DB scoring dengan flow lgbm_hyper_smote
curl "http://localhost:8000/db/score?model=lgbm_hyper_smote&snapshot_date=2026-02-06&time_range=1m"
```

## 🛠️ Persiapan dan Instalasi

### Prasyarat

- **Python 3.10+** (disarankan 3.10.11)
- **MySQL Database** dengan skema tabel: `ar_invoice_list_2`, `ar_receipt_list`, dan `OracleCustomer` (atau `OracleCustomer_slim` fallback).

### Cara Menjalankan (Dengan Docker 🐳 - Direkomendasikan)

Karena telah dilengkapi dengan `Dockerfile` dan `docker-compose.yml`, Anda dapat menjalankan API beserta seluruh komponennya dalam kontainer terisolasi:

```bash
# 1. Pastikan Docker Desktop sudah berjalan
# 2. Salin atau edit file .env dan masukkan kredensial DB
# 3. Jalankan perintah compose:
docker-compose up -d --build

# Server API akan berjalan pada http://localhost:8000
# Untuk mematikan server:
docker-compose down
```

### Cara Menjalankan (Dengan Virtual Environment / Lokal)

Jika Anda ingin melakukan proses _debugging_ atau _development_ secara langsung tanpa Docker:

````bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktivasi virtual environment
# Di Windows:
.venv\Scripts\activate
# Di Linux/Mac:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Konfigurasi Database
# Salin atau edit file .env dan masukkan kredensial:
# DB_USER=...
# DB_PASSWORD=...
# DB_HOST=...
    # DB_NAME=...

    # 5. Jalankan Server API
    python -m uvicorn api:app --host 0.0.0.0 --port 8000
    # Atau jalankan file: start_api.bat (pastikan .venv sudah aktif)
    ```

    ## 📂 Struktur Proyek Terkini

    ```text
    Model API/
    ├── api.py                      # Main Entry Point & FastAPI Endpoints
    ├── bad_debt_app/               # Core Logic & Data Layer
    │   ├── db.py                   # Konektivitas DB & Arsitektur "Two-Pass Query"
    │   └── features.py             # Feature Engineering & 16-Feature Pipeline
    ├── artifacts/                  # Model ML & Metadata
    │   ├── stacked_recall_driven_model.joblib   # Model Stacked (Utama)
    │   ├── bad_debt_snapshot_lgbm_...joblib     # Model LightGBM (Alternatif)
    │   └── feature_cols_...json                 # Daftar file schema order fitur JSON
    ├── docker-compose.yml          # Konfigurasi container komplit
    ├── Dockerfile                  # Base image dan dependencies untuk environment
    ├── .env                        # Konfigurasi Database & Global Env
    ├── start_api.bat               # Shortcut eksekusi native di Windows
    └── requirements.txt            # Daftar pustaka Python
    ```

    ### Penjelasan Detail Komponen:

    #### 1. `api.py` (FastAPI Layer)
    Mengelola seluruh _request_ dari REST Client/Frontend Next.js. Bertanggung jawab atas:
    - **Registry Model**: Memetakan kunci model (misal: `stacked`) ke file `.joblib` berserta konvensi preprocessing sklearn.
    - **Validation**: Memvalidasi target `target_trx_ids` hingga tanggal.
    - **Endpoint Routing**: Menyediakan API untuk scoring mode DB dan Mode Upload File.

    #### 2. `bad_debt_app/db.py` (Database Layer)
    Pusat komunikasi data dengan MySQL, dirancang agar tidak macet / timeout:
    - **Arsitektur "Two-Pass Query"**:
      - _Pass 1_: Menarik spesifik target waktu transaksi (menggunakan filter `1w`, `custom`, dst).
      - _Pass 2_: Mengkueri profil transaksi riewayat (*History*) hanya untuk para ID Pelanggan yang ditagihkan di periode yang relevan saja (sehingga ukuran load memori menurun drastis).
    - Berjalan dengan integrasi pada tabel khusus CM netting: `ar_invoice_list_2`.

    #### 3. `bad_debt_app/features.py` (ML Pipeline)

Jantung pengolahan data untuk 16 fitur:

- **Historical Features (9)**: Agregasi perilaku pembayaran pelanggan masa lalu (e.g., `historical_avg_dpd`, `late_payment_ratio`).
- **Survival Features (4)**: Probabilitas pembayaran dalam 30/60 hari dan perkiraan _hazard_ delay.
- **Preprocessing**: Menangani pembersihan data, _scaling_, dan konversi datetime.

#### 4. `artifacts/`

Folder penyimpanan aset statis:

- **Models**: File `.joblib` hasil training (Stacked & LightGBM).
- **Schema**: File `.json` yang memastikan urutan fitur saat kolom dikirim ke model ML tetap konsisten.

---

## 🧩 Fleksibilitas & Konfigurasi

API ini dirancang dengan tingkat fleksibilitas tinggi agar mudah beradaptasi dengan perubahan kebutuhan bisnis tanpa harus mengubah logika inti kode:

1. **Hybrid Data Source (DB vs. Upload)**
   - API mendukung pengambilan data langsung dari **Database MySQL** (untuk integrasi sistem) maupun via **Upload File CSV** (untuk keperluan audit mandiri atau simulasi data eksternal).

2. **Dynamic Model Registry**
   - Anda dapat mendaftarkan model baru (misal: versi v2, v3) cukup dengan menambahkan entitas baru pada variabel `MODEL_REGISTRY` di `api.py`. API akan otomatis mengenali model tersebut di Frontend tanpa perlu _restart_ server atau _hardcoding_ tambahan.

3. **Konfigurasi Threshold Dinamis**
   - Batas toleransi risiko (_Alert Threshold_) dapat diatur langsung melalui _query parameter_ atau UI Frontend. Ini memungkinkan tim analis untuk menyesuaikan sensitivitas peringatan dini (contoh: menurunkan threshold saat ingin lebih waspada terhadap potensi kerugian).

4. **Independent Snapshotting**
   - Pengguna bebas menentukan `snapshot_date`. Hal ini memungkinkan dilakukannya **Backtesting** (melakukan prediksi pada tanggal di masa lampau untuk melihat seberapa akurat model memprediksi _bad debt_ yang sudah terjadi).

---

## � Penyimpanan Hasil Fitur (Feature Store-ready)

Sistem ini dirancang agar hasil _Feature Engineering_ yang dihitung secara dinamis saat inferensi dapat disimpan kembali ke dalam database (**Feature Store**). Hal ini memungkinkan:

1. **Audit & Debugging**: Melacak nilai fitur pelanggan pada tanggal snapshot tertentu tanpa harus menghitung ulang dari data mentah.
2. **Monitoring Drift**: Memantau perubahan distribusi fitur (seperti `historical_avg_dpd`) dari waktu ke waktu untuk menentukan kapan model perlu dilatih ulang (_retraining_).
3. **Integrasi BI**: Tabel fitur yang tersimpan dapat langsung dihubungkan ke alat visualisasi eksternal (Tableau/PowerBI) untuk analisis risiko mendalam.

_Catatan: Implementasi penyimpanan dapat dilakukan dengan menambahkan fungsi `to_sql` pada objek DataFrame hasil `prepare_snapshot_features` di dalam `api.py`._

---

## �🛰️ Daftar API Utama (DB-Backed)

| Method  | Endpoint                            | Deskripsi                             | Parameter Utama                        |
| :------ | :---------------------------------- | :------------------------------------ | :------------------------------------- |
| **GET** | `/models`                           | Ambil daftar model & batas tanggal DB | -                                      |
| **GET** | `/db/score`                         | Prediksi risiko seluruh invoice       | `time_range`, `start_date`, `end_date` |
| **GET** | `/db/alerts`                        | Ambil hanya invoice berisiko tinggi   | `threshold` (default: 0.3)             |
| **GET** | `/db/early_warning/receipt_trigger` | Simulasi _early warning_ dari DB      | `time_range`, `start_date`, `end_date` |
| **GET** | `/db/score_csv`                     | Download hasil prediksi (CSV)         | `time_range`, `model`                  |

### Endpoint File Upload

Memerlukan unggahan file CSV invoice & receipt secara manual melalui _form data_.

| Method   | Endpoint                         | Deskripsi                            | Parameter Utama            |
| :------- | :------------------------------- | :----------------------------------- | :------------------------- |
| **POST** | `/score`                         | Upload CSV → Hasil prediksi (JSON)   | -                          |
| **POST** | `/score_csv`                     | Upload CSV → Hasil prediksi (CSV)    | -                          |
| **POST** | `/alerts`                        | Upload CSV → Filter _high-risk_ JSON | `threshold` (default: 0.3) |
| **POST** | `/early_warning/receipt_trigger` | Upload CSV → Early warning analisis  | -                          |

### Endpoint Utilities

| Method  | Endpoint  | Deskripsi                                       | Parameter Utama |
| :------ | :-------- | :---------------------------------------------- | :-------------- |
| **GET** | `/`       | Melakukan redirect otomatis ke antarmuka Web UI | -               |
| **GET** | `/health` | Mengecek status dan konfigurasi default API     | -               |

---

_Catatan: Segala perubahan pada logika fitur di `features.py` harus disinkronkan dengan schema di `feature_cols_stacked.json`._
````
