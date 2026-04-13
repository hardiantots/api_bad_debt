# Deployment Recommendations

Dokumen ini berisi saran implementasi agar publish hasil prediksi ke MySQL lebih stabil dan efisien.

## Target Operasional

1. Hasil prediksi invoice-level menjadi sumber utama di MySQL (`hasil_baddebt`).
2. `customer_risk` tetap di SQLite lokal sampai kebutuhan integrasi berikutnya.
3. Compute tidak gagal hanya karena kendala privilege DB non-kritis.

## Tahap 1: Stabilkan Mode Append-Only

1. Gunakan konfigurasi:

- `COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL=true`
- `COMPUTE_PUBLISH_TARGET_TABLE=hasil_baddebt`
- `COMPUTE_PUBLISH_REPLACE_PARTITION=false`

2. Dampak:

- Publish tidak membutuhkan `DELETE` privilege.
- Data baru pasti masuk ke MySQL selama `INSERT` privilege tersedia.

3. Risiko:

- Potensi duplikasi antar run jika partisi sama dipublish berulang.

4. Mitigasi cepat:

- Konsumen data gunakan `source_job_id` paling baru atau `published_at` terbaru.

## Tahap 2: Kurangi Duplikasi Tanpa DELETE

1. Tambahkan kunci unik di MySQL berdasarkan identitas invoice + partisi source.
2. Ubah strategi insert menjadi upsert (`INSERT ... ON DUPLICATE KEY UPDATE`) bila diperlukan.
3. Alternatif: jalankan proses dedup terjadwal di sisi DBA.

## Tahap 3: Aktifkan Replace Partition Penuh

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

- jumlah row compute lokal (`bad_debt_score_results`)
- jumlah row masuk MySQL (`hasil_baddebt`)
- selisih jumlah row per `source_job_id`

3. Alert sederhana:

- jika compute completed tapi tidak ada row MySQL untuk `source_job_id` tersebut.

## Prosedur Incident (Publish Gagal)

1. Cek privilege DB user.
2. Cek keberadaan tabel target dan kolom mapping.
3. Jalankan publish manual endpoint:

- `POST /db/compute/publish?job_id=<JOB_ID>&table_name=hasil_baddebt&replace_partition=false`

4. Jika publish manual sukses, fokus perbaikan pada mode auto-publish/environment.

## Checklist Rilis

1. PM2 status `online`.
2. Endpoint `/health` respon normal.
3. Compute test run sukses.
4. Row prediksi masuk ke `hasil_baddebt`.
5. Tidak ada error berulang pada log publish.
