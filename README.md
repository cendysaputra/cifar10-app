# cifar10-app

Aplikasi web untuk mengklasifikasikan satu objek pada gambar ke dalam 10 kelas CIFAR-10: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, dan Truck.

Model yang dipakai adalah CNN kecil bergaya ResNet (`model_cnn_cifar10_v4.h5`). Pada data uji CIFAR-10, model v4 ini dilaporkan mencapai akurasi 91,24%. Foto biasa tidak akan setepat itu: sebelum diprediksi, gambar ditegakkan menurut data EXIF, di-crop persegi dari tengah jika tidak persegi, lalu diperkecil menjadi 32×32 piksel dengan nilai warna 0–1.

Aplikasi dibuat dengan Streamlit dan TensorFlow/Keras.

## Struktur Proyek

- `app.py` — antarmuka Streamlit
- `classify.py` — pembacaan gambar, praproses, dan ringkasan prediksi
- `style.css` — tampilan hasil prediksi
- `model_cnn_cifar10_v4.h5` — model yang dipakai aplikasi
- `train.py` — pelatihan ulang model
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

Buka alamat yang dicetak di terminal, lalu unggah file JPG, PNG, atau WEBP (maksimal 5 MB). Jika keyakinan model di bawah 50%, halaman menampilkan peringatan.

## Uji

```bash
pip install -r requirements-dev.txt
pytest
```

Uji model memuat berkas `.h5` yang ada di repositori dan memeriksa bahwa keluarannya 10 probabilitas. Uji ini tidak melatih ulang model.

## Melatih Ulang Model

Model dilatih otomatis oleh GitHub Actions setiap kali `train.py` atau workflow pelatihannya berubah di branch selain `main`, atau secara manual lewat **Actions → Train model → Run workflow**. Model hasil training di-commit kembali ke branch yang sama.

Pelatihan lokal:

```bash
pip install -r requirements.txt
python train.py --epochs 60
```

Perintah itu mengunduh CIFAR-10 lewat Keras dan menulis `model_cnn_cifar10_v4.h5`.

## Deployment

Aplikasi di-deploy lewat Streamlit Community Cloud dari repositori ini.

- Berkas utama: `app.py`
- Versi Python: `runtime.txt` (3.11)
- Dependensi: `requirements.txt`
- Unggahan dibatasi 5 MB di `.streamlit/config.toml`

Berkas model harus tetap ada di repositori agar Cloud dapat memuatnya. Jangan menyimpan rahasia di kode; jika suatu saat diperlukan, gunakan `.streamlit/secrets.toml`, yang tidak di-commit.
