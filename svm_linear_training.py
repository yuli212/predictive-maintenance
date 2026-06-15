import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

print("=== EVALUASI SVM KERNEL LINEAR (BUKTI KEGAGALAN) ===\n")

# --- 1. MEMUAT DATA HASIL PCA ---
try:
    df = pd.read_excel("Dataset_Hasil_PCA.xlsx")
except FileNotFoundError:
    print("[ERROR] File Dataset_Hasil_PCA.xlsx tidak ditemukan!")
    exit()

X = df[['PC1', 'PC2']].values
y = df['Kondisi'].map({'Normal': 0, 'Indikasi Kotor': 1}).values

# --- 2. STANDARISASI FITUR ---
# Wajib dilakukan agar skala PC1 dan PC2 seimbang
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PELATIHAN & VALIDASI (KERNEL LINEAR) ---
print("Menjalankan 5-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inisialisasi model dengan kernel linear
model_linear = SVC(kernel='linear', class_weight='balanced', C=1.0)

# Uji performa
skor_akurasi = cross_val_score(model_linear, X_scaled, y, cv=kf, scoring='accuracy')
print(f"-> Akurasi per Fold: {[round(acc*100, 2) for acc in skor_akurasi]}")
print(f"-> RATA-RATA AKURASI: {np.mean(skor_akurasi)*100:.2f}%\n")

# Melatih model menggunakan seluruh data untuk metrik final
model_linear.fit(X_scaled, y)
y_pred = model_linear.predict(X_scaled)

print("Laporan Klasifikasi Final:")
print(classification_report(y, y_pred, target_names=['Normal', 'Indikasi Kotor']))

# --- 4. VISUALISASI HYPERPLANE LINEAR ---
plt.figure(figsize=(9, 6))
fig_manager = plt.get_current_fig_manager()
try:
    fig_manager.set_window_title('Kegagalan SVM Linear')
except AttributeError:
    pass # Kompatibilitas untuk beberapa versi matplotlib backend

# Memisahkan titik data untuk diplot
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], 
            color='blue', label='Normal', edgecolors='k', alpha=0.7)
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
            color='red', label='Indikasi Kotor', edgecolors='k', alpha=0.5)

# Menggambar batas keputusan (garis lurus)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Menghitung jarak keputusan linear
Z = model_linear.decision_function(xy).reshape(XX.shape)

# Menggambar contour margin dan hyperplane
ax.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

plt.title('Batas Keputusan SVM (Kernel: Linear)\nKarakteristik: Gagal Mengisolasi Pola Elips', 
          fontsize=13, fontweight='bold', color='red')
plt.xlabel('PC1 (Standardized)')
plt.ylabel('PC2 (Standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()