# Deployment Recommendations

Dokumen ini berisi saran operasional agar alur hasil prediksi (invoice-level) MySQL-first tetap stabil dan efisien.

## Target Operasional

1. Arsitektur Pure REST: Semua endpoint yang menyajikan data berjalan secara `Read-Only`. Pemrosesan scoring model hanya terjadi lewat permintaan eksplisit dari Front-End (via _Refresh Scoring_) atau penjadwalan via _Background Cron Job_.
2. Optimalisasi Dynamic Time Filter: Selalu jadwalkan komputasi untuk range waktu yang paling panjang (misal `3m` atau `6m`). Hasil ini akan melayani permintaan API filter rentang waktu pendek secara efisien melalui kloning subset database.
3. Hasil prediksi invoice-level menjadi sumber utama integrasi dan disajikan langsung dari MySQL (`hasil_baddebt`); pastikan UUID string `job_id` tidak terurai menjadi `NULL` saat pengisian data.
4. Struktur `customer_risk` (agregasi pelanggan) tetap terpusat di SQLite lokal (`scoring.db`) sampai kebutuhan integrasi lintas DB berikutnya memintanya dipindah.
5. Komputasi model (compute payload) tidak gagal secara keseluruhan di tahap akhir publikasi hanya karena user database tidak memiliki hak `DELETE` untuk mereplace partisi.

## Mode Saat Ini (Disarankan)

1. Gunakan konfigurasi:

- `COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL=true`
- `COMPUTE_PUBLISH_TARGET_TABLE=hasil_baddebt`
- `COMPUTE_PUBLISH_REPLACE_PARTITION=false`

2. Dampak:

- Publish tidak membutuhkan `DELETE` privilege.
- Data baru pasti masuk ke MySQL selama `INSERT` privilege tersedia.
- Endpoint prediksi membaca dari MySQL table (`hasil_baddebt`).

3. Risiko:

- Potensi duplikasi antar run jika partisi sama dipublish berulang.

4. Mitigasi cepat:

- Konsumen data gunakan `source_job_id` paling baru atau `published_at` terbaru.

## Penguatan Tahap Lanjut

1. Tambahkan kunci unik di MySQL berdasarkan identitas invoice + partisi source.
2. Ubah strategi insert menjadi upsert (`INSERT ... ON DUPLICATE KEY UPDATE`) bila diperlukan.
3. Alternatif: jalankan proses dedup terjadwal di sisi DBA.

## Opsi Replace Partition Penuh

1. Saat privilege sudah memungkinkan, aktifkan:

- `COMPUTE_PUBLISH_REPLACE_PARTITION=true`

2. Pastikan user DB punya privilege:

- `SELECT`, `INSERT`, `DELETE` pada tabel `hasil_baddebt`.

3. Dampak:

- Data partisi sama bisa ditimpa bersih pada setiap compute.

## Monitoring Minimal

1. Pantau log aplikasi:

- keyword `Auto-published score`
- keyword `Auto-publish score failed`

2. Pantau metrik harian:

- jumlah row masuk MySQL (`hasil_baddebt`) per `source_job_id`
- jumlah row customer risk lokal per `job_id`
- selisih row antar job untuk partisi yang sama (indikasi duplikasi)

3. Alert sederhana:

- jika compute completed tapi tidak ada row MySQL untuk `source_job_id` tersebut.

## Prosedur Incident (Publish Gagal)

1. Cek privilege DB user.
2. Cek keberadaan tabel target dan kolom mapping.
3. Jalankan compute ulang untuk partisi yang bermasalah:

- `POST /db/compute?model=<MODEL>&snapshot_date=<YYYY-MM-DD>&time_range=<RANGE>`

4. Verifikasi log `Auto-published score` untuk `job_id` terbaru.

5. Jika error muncul pada endpoint `GET /db/customer_risk` dengan pesan SQLite tidak bisa dibuka:

- pastikan direktori `local_data/` ada
- pastikan permission write sesuai user proses PM2
- restart service setelah permission diperbaiki

## Checklist Rilis

1. PM2 status `online`.
2. Endpoint `/health` respon normal.
3. Compute test run sukses.
4. Row prediksi masuk ke `hasil_baddebt`.
5. Tidak ada error berulang pada log publish.
