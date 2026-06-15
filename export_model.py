import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from micromlgen import port

try:
    df = pd.read_excel("Dataset_Hasil_PCA.xlsx")
except FileNotFoundError:
    print("File Dataset_Hasil_PCA.xlsx tidak ditemukan")
    exit()

X = df[['PC1', 'PC2']].values
y = df['Kondisi'].map({'Normal': 0, 'Indikasi Kotor': 1}).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# gamma=1.0 agar micromlgen bisa ekstrak array C++ dengan benar
model_rbf = SVC(kernel='rbf', class_weight='balanced', C=50.0, gamma=1.0)
model_rbf.fit(X_scaled, y)

print("Mengekspor model ke C++...")
c_header = port(model_rbf, classmap={0: 'Normal', 1: 'Indikasi_Kotor'})

folder_tujuan = "hardware"
os.makedirs(folder_tujuan, exist_ok=True)
nama_file = os.path.join(folder_tujuan, "SvmRbfModel.h")

with open(nama_file, "w") as f:
    f.write(c_header)

print(f"Selesai, file disimpan ke: {nama_file}")

# parameter scaler untuk arduino
print("\nParameter StandardScaler untuk ESP32:")
print(f"float mean_pc1 = {scaler.mean_[0]:.6f};")
print(f"float mean_pc2 = {scaler.mean_[1]:.6f};")
print(f"float scale_pc1 = {scaler.scale_[0]:.6f};")
print(f"float scale_pc2 = {scaler.scale_[1]:.6f};")
