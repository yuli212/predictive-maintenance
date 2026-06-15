import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from micromlgen import port

print("=== FASE 4: EKSPOR SVM RBF KE FOLDER HARDWARE (FIX GAMMA) ===\n")

# --- 1. MEMUAT DATA ---
try:
    df = pd.read_excel("Dataset_Hasil_PCA.xlsx")
except FileNotFoundError:
    print("[ERROR] File Dataset_Hasil_PCA.xlsx tidak ditemukan di folder utama!")
    exit()

X = df[['PC1', 'PC2']].values
y = df['Kondisi'].map({'Normal': 0, 'Indikasi Kotor': 1}).values

# --- 2. STANDARISASI ---
print("Menstandarisasi data dan melatih model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PELATIHAN MODEL SVM RBF (DENGAN GAMMA EKSPLISIT) ---
# Menggunakan gamma=1.0 agar micromlgen sukses melakukan ekstraksi array C++
model_rbf = SVC(kernel='rbf', class_weight='balanced', C=50.0, gamma=1.0)
model_rbf.fit(X_scaled, y)

# --- 4. EKSPOR MODEL MENGGUNAKAN MICROMLGEN ---
print("Menerjemahkan model ke bahasa C++...")
# classmap memastikan output di C++ nanti langsung berupa teks
c_header = port(model_rbf, classmap={0: 'Normal', 1: 'Indikasi_Kotor'})

# --- 5. MANAJEMEN DIREKTORI & PENYIMPANAN ---
folder_tujuan = "hardware"
os.makedirs(folder_tujuan, exist_ok=True)
nama_file = os.path.join(folder_tujuan, "SvmRbfModel.h")

with open(nama_file, "w") as f:
    f.write(c_header)

print("-" * 60)
print(f"[SUKSES] Model SVM RBF berhasil diekspor ke: ./{nama_file}")
print("-" * 60)

# --- 6. PARAMETER STANDARISASI UNTUK C++ ---
print("\n[CATATAN PENTING] Parameter StandardScaler untuk kodingan Arduino:")
print("Data sensor MPU6050 Anda harus dinormalisasi dengan angka ini")
print("sebelum diumpankan ke model prediksi SVM di ESP32:\n")

print(f"float mean_pc1 = {scaler.mean_[0]:.6f};")
print(f"float mean_pc2 = {scaler.mean_[1]:.6f};")
print(f"float scale_pc1 = {scaler.scale_[0]:.6f};")
print(f"float scale_pc2 = {scaler.scale_[1]:.6f};")