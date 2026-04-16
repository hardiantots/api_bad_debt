# Deployment Recommendations

Dokumen ini berisi saran operasional agar alur hasil prediksi (invoice-level) MySQL-first tetap stabil dan efisien.

## Target Operasional

1. Arsitektur Pure REST: Semua endpoint yang menyajikan data berjalan secara `Read-Only`. Pemrosesan scoring model hanya terjadi lewat permintaan eksplisit dari Front-End (via _Refresh Scoring_) atau penjadwalan via _Background Cron Job_.
2. Optimalisasi Dynamic Time Filter: Selalu jadwalkan komputasi untuk range waktu yang paling panjang (misal `3m` atau `6m`). Hasil ini akan melayani permintaan API filter rentang waktu pendek secara efisien melalui kloning subset database.
3. Hasil prediksi invoice-level menjadi sumber utama integrasi dan disajikan langsung dari MySQL (`hasil_baddebt`); pastikan UUID string `job_id` tidak terurai menjadi `NULL` saat pengisian data.
4. Struktur `customer_risk` (agregasi pelanggan) kini ikut terpusat dan dipublikasikan ke MySQL di tabel terpisah (`customer_risk`), sehingga seluruh data scoring terintegrasi di database yang sama.
5. Komputasi model (compute payload) kini membutuhkan user database yang memiliki hak `DELETE` untuk mereplace partisi secara bersih.

## Mode Saat Ini (Aktif & Disarankan)

1. Gunakan konfigurasi:

- `COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL=true`
- `COMPUTE_PUBLISH_TARGET_TABLE=hasil_baddebt`
- `COMPUTE_PUBLISH_REPLACE_PARTITION=true`

2. Dampak:

- Publish sekarang membutuhkan `DELETE` privilege pada database user.
- Data baru di-insert dan data lama untuk partisi yang sama ditimpa bersih.
- Endpoint prediksi membaca langsung dari tabel MySQL (`hasil_baddebt` dan tabel baru `customer_risk`).

3. Risiko:

- Jika delete gagal tanpa transaksi utuh, berpotensi data hilang sesaat untuk partisi tersebut.

4. Mitigasi cepat:

- Konsumen data gunakan `source_job_id` paling baru atau `published_at` terbaru.

## Penguatan Tahap Lanjut

1. Tambahkan kunci unik di MySQL berdasarkan identitas invoice + partisi source.
2. Gunakan query yang disesuaikan dalam dashboard untuk mengatasi potensi asinkron pembaruan dari kedua tabel agregat MySQL.
3. Alternatif: siapkan sistem alarm / alert jika data model out of sync dengan MySQL.

## Monitoring Minimal

1. Pantau log aplikasi:

- keyword `Auto-published score`
- keyword `Auto-publish score failed`

2. Pantau metrik harian:

- jumlah row masuk MySQL (`hasil_baddebt`) per `source_job_id`
- jumlah row tabel agregasi MySQL (customer risk) per `job_id`
- log durasi untuk penghapusan partisi lama (replace partition)

3. Alert sederhana:

- jika compute completed tapi tidak ada row MySQL untuk `source_job_id` tersebut.

## Prosedur Incident (Publish Gagal)

1. Cek privilege DB user.
2. Cek keberadaan tabel target dan kolom mapping.
3. Jalankan compute ulang untuk partisi yang bermasalah:

- `POST /db/compute?model=<MODEL>&snapshot_date=<YYYY-MM-DD>&time_range=<RANGE>`

4. Verifikasi log `Auto-published score` untuk `job_id` terbaru.

5. Jika error muncul pada endpoint `GET /db/customer_risk` dengan pesan koneksi MySQL terputus atau tabel tidak ditemukan:

- pastikan tabel risk pelanggan sudah ter-deploy di database
- perhatikan log privilege terkait apakah user memiliki `SELECT` dan juga `DELETE` (saat update via compute)
- restart service jika konfigurasi db driver bermasalah

## Checklist Rilis

1. PM2 status `online`.
2. Endpoint `/health` respon normal.
3. Compute test run sukses.
4. Row prediksi masuk ke `hasil_baddebt`.
5. Tidak ada error berulang pada log publish.
