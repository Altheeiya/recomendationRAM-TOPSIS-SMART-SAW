# SPK Pemilihan RAM Terbaik — Streamlit App

Aplikasi Sistem Penunjang Keputusan (SPK) untuk memilih RAM terbaik
berdasarkan data benchmark DDR2/DDR3/DDR4/DDR5 menggunakan metode
SMART, SAW, dan TOPSIS.

---



## Cara Pakai Aplikasi

1. **Download dataset** dari Kaggle:
   https://www.kaggle.com/datasets/alanjo/ddr2ddr3ddr4ddr5-ram-benchmarks
   File yang dipakai: `RAM_Benchmarks_megalist.csv`

2. **Upload CSV** di sidebar kiri aplikasi

3. **Atur filter** di sidebar:
   - Pilih generasi RAM (DDR2/DDR3/DDR4/DDR5)
   - Tentukan jumlah alternatif
   - Atur bobot kriteria (total harus = 1.0)

4. **Lihat hasil** di tab masing-masing metode

---

## Kolom Dataset

| Kolom        | Deskripsi             | Tipe Kriteria |
|--------------|-----------------------|---------------|
| memoryName   | Nama produk RAM       | Identifier    |
| gen          | Generasi (DDR2–DDR5)  | Filter        |
| latency      | Latency (ns)          | Cost          |
| readUncached | Read speed (MB/s)     | Benefit       |
| write        | Write speed (MB/s)    | Benefit       |
| price        | Harga (USD)           | Cost          |

---

## Metode SPK

### SMART (Simple Multi-Attribute Rating Technique)
Normalisasi nilai ke skala utilitas 0–100, lalu dikalikan bobot.
- Benefit: `u(xi) = (xi - min) / (max - min) × 100`
- Cost: `u(xi) = (max - xi) / (max - min) × 100`

### SAW (Simple Additive Weighting)
Normalisasi relatif terhadap nilai terbaik di setiap kriteria.
- Benefit: `rij = xij / max(xj)`
- Cost: `rij = min(xj) / xij`

### TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
Mempertimbangkan jarak ke solusi ideal positif (A+) dan negatif (A-).
- `Ci = D- / (D+ + D-)`

---

## Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
```
