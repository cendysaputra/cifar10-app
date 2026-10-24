# cifar10-app

Aplikasi web sederhana untuk klasifikasi gambar menggunakan model CNN yang dilatih pada dataset CIFAR-10. Pengguna dapat mengunggah gambar, lalu model akan memprediksi kelas objek dari 10 kelas yang tersedia: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, dan Truck.

Aplikasi dibangun dengan Streamlit dan menggunakan model TensorFlow/Keras.

## Struktur Proyek

- `app.py` — kode utama aplikasi Streamlit
- `requirements.txt` — daftar dependensi
- `model_cnn_cifar10_v3.h5` — model CNN hasil pelatihan
- `train.py` — script untuk melatih ulang model
- `.github/workflows/train-model.yml` — workflow GitHub Actions untuk training otomatis
- `README.md` — dokumentasi proyek

## Menjalankan Secara Lokal

Pasang dependensi:

```
pip install -r requirements.txt
```

Jalankan aplikasi:

```
streamlit run app.py
```

## Melatih Ulang Model

Model dilatih otomatis oleh GitHub Actions setiap kali `train.py` berubah, atau manual lewat tab **Actions → Train model → Run workflow**. Model hasil training di-commit ke branch yang sama sebagai `model_cnn_cifar10_v3.h5`.

Training juga bisa dijalankan secara lokal:

```
pip install tensorflow-cpu==2.21.0
python train.py --epochs 40
```

## Deployment

Aplikasi di-deploy melalui Streamlit Community Cloud yang terhubung dengan repository ini.
