import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

print("Memulai Fase 3 (Revisi): Pelatihan SVM yang Dioptimalkan...\n")

# --- 1. MEMUAT DATA HASIL PCA ---
try:
    df = pd.read_excel("Dataset_Hasil_PCA.xlsx")
except FileNotFoundError:
    print("[ERROR] File Dataset_Hasil_PCA.xlsx tidak ditemukan!")
    exit()

X = df[['PC1', 'PC2']].values
y = df['Kondisi'].map({'Normal': 0, 'Indikasi Kotor': 1}).values

# --- 2. MODIFIKASI 1: STANDARISASI ULANG KOMPONEN PCA ---
# Menyeimbangkan rentang PC1 dan PC2 agar SVM tidak bias pada varians terbesar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. MODIFIKASI 2: PENYEIMBANG KELAS DI K-FOLD ---
print("Menjalankan 5-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Menambahkan class_weight='balanced' agar tarikan garis batas berada persis di tengah
model_svm = SVC(kernel='rbf', class_weight='balanced', C=50.0, gamma='scale')

skor_akurasi = cross_val_score(model_svm, X_scaled, y, cv=kf, scoring='accuracy')
print(f"-> Akurasi per Fold: {[round(acc*100, 2) for acc in skor_akurasi]}")
print(f"-> RATA-RATA AKURASI: {np.mean(skor_akurasi)*100:.2f}%\n")

# --- 4. PELATIHAN MODEL FINAL & MATRIKS EVALUASI ---
model_svm.fit(X_scaled, y)
y_pred = model_svm.predict(X_scaled)

print("Laporan Klasifikasi Final:")
print(classification_report(y, y_pred, target_names=['Normal', 'Indikasi Kotor']))

# --- 4. EKSTRAKSI BOBOT UNTUK ESP32 (DI-NONAKTIFKAN KARENA KERNEL RBF) ---
print("-" * 50)
print("=== CATATAN UNTUK LAPORAN ===")
print("Kernel RBF menggunakan Support Vectors, bukan bobot linear.")
print(f"Total Support Vectors yang terbentuk: {len(model_svm.support_vectors_)} titik.")
print("-" * 50)

# --- 6. VISUALISASI HYPERPLANE SVM YANG BARU ---
plt.figure(figsize=(10, 6))

# Plot titik data yang sudah diskala ulang
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='blue', label='Normal', edgecolors='k', alpha=0.7)
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='red', label='Indikasi Kotor', edgecolors='k', alpha=0.7)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Memperhalus resolusi grid untuk gambar garis (dari 30 menjadi 100)
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_svm.decision_function(xy).reshape(XX.shape)

# Menggambar margin dan batas keputusan
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

plt.title('Batas Keputusan SVM (Hyperplane) - Pasca Optimasi Skala', fontweight='bold')
plt.xlabel('PC1 (Standardized)')
plt.ylabel('PC2 (Standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()