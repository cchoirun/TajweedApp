Aplikasi deteksi hukum tajwid

---

## Isi aplkikasi

- Upload file audio
- Visualisasi waveform
- Detail prediksi dengan confidence

## Persiapan model
masukkan model yang mau dipakai didalam folder models

## Instalasi
### 0. Opsional: install virtual environment (kalau misal error jika tanpa venv)
```bash
py -m venv venv
```
Untuk masuk ke venv:
```bash
venv\Scripts\Activate
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
python app.py
```

Buka browser: **http://localhost:5000**

---


### Tambahan dependensi: (jika muncul error)

```bash
pip install imageio-ffmpeg
winget install ffmpeg
pip install ffmpeg-python
````
