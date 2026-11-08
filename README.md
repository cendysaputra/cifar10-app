# cifar10-app

Aplikasi web untuk mengklasifikasikan satu foto. Ada dua mode:

- **CIFAR-10** — model CNN yang sudah ada, 10 kelas: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, dan Truck.
- **Umum** — model zero-shot untuk benda sehari-hari. Daftarnya 256 label, termasuk orang, rumah, batu, hewan, makanan, kendaraan, dan pemandangan.

## Mode CIFAR-10

Modelnya ResNet-18 untuk CIFAR (`model_cnn_cifar10_v5.h5`, 11,17 juta parameter, 45 MB). Pada 10.000 gambar uji CIFAR-10 resmi, diukur sekali setelah pelatihan selesai:

- **95,71%** satu tampilan (9.571/10.000)
- **96,06%** rata-rata dengan cermin horizontal (9.606/10.000). Aplikasi memakai angka ini.

Foto biasa tidak akan setepat itu: sebelum diprediksi, gambar ditegakkan menurut data EXIF, di-crop persegi dari tengah jika tidak persegi, lalu diperkecil menjadi 32×32 piksel dengan nilai warna 0–1. Model menormalisasi warna di dalamnya. Prediksi merata-ratakan gambar dengan cermin horizontalnya. Jika probabilitas tertinggi di bawah 50%, halaman menampilkan peringatan.

Satu gambar di mesin latihan memakan sekitar 38 ms, atau 46 ms untuk dua tampilan (cermin). Model v4 sebelumnya sekitar 26 ms untuk satu tampilan.

### Cara angka uji didapat

Himpunan latih resmi 50.000 gambar dibagi dengan seed 42: 45.000 untuk pelatihan dan 5.000 untuk validasi (500 gambar per kelas). Himpunan uji tidak dipakai untuk melatih atau memilih checkpoint. Checkpoint terbaik adalah epoch 196 menurut akurasi validasi dengan cermin horizontal (96,48% pada 5.000 gambar validasi; 96,24% tanpa cermin). Angka itu bukan akurasi uji.

Resepnya: pad 4 lalu crop acak, flip horizontal, Cutout 16, SGD Nesterov momentum 0,9, weight decay 5e-4, batch 128, 200 epoch, learning rate 0,1 yang turun 5× setelah epoch 60, 120, dan 160. Mesin ini tidak punya GPU. Pelatihan memakai 4 inti CPU dengan komputasi bfloat16 dan bobot float32, selama 23.566,9 detik (6,55 jam). Ekspor Keras dicek terhadap PyTorch sampai selisih probabilitas di bawah 1e-6.

Bukti yang di-commit:

- [Log pelatihan lengkap](results/train_log.txt), termasuk evaluasi uji di akhir
- [Metrik JSON](results/metrics.json): akurasi uji, per kelas, epoch, seed, perangkat, tanggal
- [Matriks konfusi uji, cermin horizontal](results/confusion_matrix_tta.png)
- [SHA-256 model](results/model_sha256.txt): `539a2abe2e9145eeaff34e21a295d77bf12c4db5f0967143db98467b08d52b71`

Mengukur ulang berkas yang sama, tanpa melatih:

```bash
python scripts/eval_cifar_test.py
```

Perintah itu mencetak 9.571/10.000 dan 9.606/10.000. `--write-results` menulis ulang JSON, gambar, dan checksum.

Mode ini hanya memuat TensorFlow saat dipilih.

## Mode Umum

Mode Umum memakai CLIP ViT-B/32 dari OpenAI, dalam berkas ONNX terkuantisasi (sekitar 85 MB) lewat ONNX Runtime. Bobot tidak disimpan di git. Kunjungan pertama mengunduhnya ke cache server, lalu `st.cache_resource` menahan session-nya. Embedding teks untuk 256 label sudah dihitung dan ikut di repositori (`data/general_label_embeddings.npz`, sekitar 575 KB), jadi runtime hanya menjalankan encoder gambar.

Label ditulis dalam bahasa Inggris untuk CLIP, lalu ditampilkan dalam bahasa Indonesia. Contoh: `house` menjadi Rumah, `rock` menjadi Batu (prompt-nya mencakup batu, stone, dan boulder), `man` menjadi Pria.

Hasilnya 5 tebakan teratas beserta probabilitas. Probabilitas itu softmax atas ratusan label, jadi tebakan yang benar sering berada di bawah 50%. Halaman menulis **Tidak yakin (not sure)** jika foto tidak lebih mirip label mana pun dibanding kalimat generik ("gambar acak", "gambar abstrak", "foto buram"), atau jika dua label teratas hampir seri.

Batasan yang memang ada:

- Ini bukan detektor yang dilatih pada rumah atau orang. Foto yang jelas biasanya kena, foto yang sibuk bisa meleset.
- Batu yang memenuhi bingkai terbaca sebagai Batu. Foto yang separuh air bisa terbaca sebagai danau, atau ditandai tidak yakin.
- Orang dengan aksesori yang sangat menonjol, misalnya kacamata di cahaya latar, bisa ditandai tidak yakin.
- Daftar label tertutup. Benda di luar daftar akan menempel ke label terdekat atau ditandai tidak yakin.

Pada mesin uji, session CLIP memakai sekitar 200 MB RAM setelah Streamlit dimuat. Memuat model CIFAR-10 memakai sekitar 600 MB karena TensorFlow. Aplikasi hanya memuat model mode yang sedang aktif. Di paket gratis Streamlit Community Cloud (sekitar 1 GB RAM), muat ulang halaman saat berganti mode: TensorFlow sering tidak mengembalikan memorinya ke sistem operasi.

## Struktur Proyek

- `app.py` — antarmuka Streamlit dan sakelar mode
- `classify.py` — pembacaan gambar, praproses CIFAR-10, dan ringkasan prediksi
- `general.py` — unduh model Umum, praproses CLIP, dan skor label
- `general_labels.py` — 256 label, prompt bahasa Inggris, nama tampilan bahasa Indonesia
- `data/general_label_embeddings.npz` — embedding teks yang sudah dihitung
- `scripts/build_general_embeddings.py` — bangun ulang embedding itu (bukan dependensi runtime)
- `style.css` — tampilan hasil prediksi
- `model_cnn_cifar10_v5.h5` — model CIFAR-10 yang dipakai aplikasi
- `train.py` — pelatihan ulang model
- `requirements-train.txt` — PyTorch CPU, hanya untuk pelatihan
- `scripts/eval_cifar_test.py` — ukur ulang akurasi uji pada berkas model
- `results/` — log pelatihan, metrik, matriks konfusi, dan SHA-256 model
- `requirements.txt` — dependensi aplikasi
- `requirements-dev.txt` — dependensi aplikasi plus pytest
- `runtime.txt` — Python 3.11 untuk Streamlit Community Cloud
- `.streamlit/config.toml` — batas unggah 5 MB dan tema terang
- `tests/` — uji praproses, aplikasi, dan keluaran model
- `.github/workflows/test.yml` — uji otomatis pada pull request
- `.github/workflows/train-model.yml` — pelatihan ulang di GitHub Actions

## Menjalankan Secara Lokal

Gunakan Python 3.11, sesuai `runtime.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang dicetak di terminal, lalu pilih mode dan unggah file JPG, PNG, atau WEBP (maksimal 5 MB). Mode CIFAR-10 langsung memakai berkas `.h5`. Mode Umum mengunduh model CLIP sekali, sekitar 85 MB, lalu memakainya dari cache.

## Uji

```bash
pip install -r requirements-dev.txt
pytest
```

Uji model CIFAR memuat berkas `.h5` yang ada di repositori dan memeriksa bahwa keluarannya 10 probabilitas. Uji ini tidak melatih ulang model.

Uji mode Umum yang biasa tidak mengunduh apa pun: skornya diuji dengan embedding palsu dan dengan vektor yang sudah disimpan. Foto sungguhan (orang, rumah, batu) diuji terpisah, karena itu mengunduh atau membaca model CLIP:

```bash
GENERAL_LIVE=1 pytest tests/test_general_live.py
```

Untuk membangun ulang embedding setelah mengubah `general_labels.py`:

```bash
pip install onnxruntime tokenizers numpy
python scripts/build_general_embeddings.py
```

## Melatih Ulang Model

Pelatihan 200 epoch tidak muat di batas waktu runner GitHub, jadi workflow **Actions → Train model → Run workflow** hanya dijalankan manual dan berhenti sesuai batas waktu. Model yang ada di repositori dilatih dengan skrip yang sama di luar runner.

```bash
pip install -r requirements.txt -r requirements-train.txt
python train.py --epochs 200
```

Perintah itu mengunduh CIFAR-10, menahan 5.000 gambar latih sebagai validasi, dan menulis `model_cnn_cifar10_v5.h5`. Himpunan uji baru dibaca setelah jadwal selesai.

## Deployment

Aplikasi di-deploy lewat Streamlit Community Cloud dari repositori ini.

- Berkas utama: `app.py`
- Versi Python: `runtime.txt` (3.11)
- Dependensi: `requirements.txt` (TensorFlow untuk CIFAR-10, ONNX Runtime untuk mode Umum)
- Unggahan dibatasi 5 MB di `.streamlit/config.toml`

Berkas `model_cnn_cifar10_v5.h5` harus tetap ada di repositori agar mode CIFAR-10 dapat memuatnya. Repositori ini tidak memakai Git LFS; berkas model 45 MB masih di bawah batas 100 MB GitHub. Bobot CLIP tidak di-commit; server mengunduhnya dari Hugging Face saat mode Umum pertama kali dipakai. Jangan menyimpan rahasia di kode; jika suatu saat diperlukan, gunakan `.streamlit/secrets.toml`, yang tidak di-commit.
