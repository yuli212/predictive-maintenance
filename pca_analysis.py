import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Library khusus untuk membuat grafik 3 Dimensi
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

print("Memulai Modifikasi Tahap 2: PCA dengan Ekspor File dan Visualisasi Komparasi...\n")

# --- 1. MEMUAT DATA (Fokus pada Accel X, Y, Z) ---
try:
    df_normal = pd.read_excel("IMU_Normal.xlsx")
    df_kotor = pd.read_excel("IMU_Kotor.xlsx")
except FileNotFoundError:
    print("[ERROR] File Excel tidak ditemukan. Pastikan Anda berada di folder yang tepat.")
    exit()

df_normal['Label'] = 0
df_kotor['Label'] = 1
df_gabungan = pd.concat([df_normal, df_kotor], ignore_index=True)

kolom_fitur = ['Accel X (g)', 'Accel Y (g)', 'Accel Z (g)']
X = df_gabungan[kolom_fitur]
y = df_gabungan['Label']

# --- 2. STANDARISASI ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. REDUKSI DIMENSI (PCA) ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Menyusun DataFrame hasil reduksi
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Label'] = y
# Mengubah label angka jadi teks agar mudah dibaca di file Excel nanti
df_pca['Kondisi'] = df_pca['Label'].map({0: 'Normal', 1: 'Indikasi Kotor'})

# --- FITUR BARU 1: EKSPOR HASIL PCA KE FILE EXCEL ---
nama_file_output = "Dataset_Hasil_PCA.xlsx"
# Kita simpan PC1, PC2, dan Kondisinya
df_pca[['PC1', 'PC2', 'Kondisi']].to_excel(nama_file_output, index=False)

print(f"[BERHASIL] Data hasil reduksi PCA telah disimpan ke: {nama_file_output}")
print(f"Dimensi Awal   : {X.shape[0]} baris x {X.shape[1]} sumbu fisis")
print(f"Dimensi Output : {df_pca.shape[0]} baris x 2 komponen utama\n")

# --- FITUR BARU 2: VISUALISASI SEBELUM DAN SESUDAH REDUKSI ---
# Membuat jendela grafik memanjang untuk 2 diagram
fig = plt.figure(figsize=(15, 6))
fig.canvas.manager.set_window_title('Komparasi PCA - Pattern Recognition')

# [Grafik Kiri] SEBELUM REDUKSI (Plot 3D)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X.loc[y==0, 'Accel X (g)'], X.loc[y==0, 'Accel Y (g)'], X.loc[y==0, 'Accel Z (g)'], 
            c='blue', label='Normal', alpha=0.6, s=20)
ax1.scatter(X.loc[y==1, 'Accel X (g)'], X.loc[y==1, 'Accel Y (g)'], X.loc[y==1, 'Accel Z (g)'], 
            c='red', label='Indikasi Kotor', alpha=0.6, s=20)

ax1.set_title("SEBELUM PCA\n(Ruang Fisis 3 Dimensi)", fontweight='bold')
ax1.set_xlabel('Akselerasi X (g)')
ax1.set_ylabel('Akselerasi Y (g)')
ax1.set_zlabel('Akselerasi Z (g)')
ax1.legend()

# [Grafik Kanan] SESUDAH REDUKSI (Plot 2D)
ax2 = fig.add_subplot(122)
ax2.scatter(df_pca.loc[y==0, 'PC1'], df_pca.loc[y==0, 'PC2'], 
            c='blue', label='Normal', alpha=0.7, edgecolors='k', s=40)
ax2.scatter(df_pca.loc[y==1, 'PC1'], df_pca.loc[y==1, 'PC2'], 
            c='red', label='Indikasi Kotor', alpha=0.7, edgecolors='k', s=40)

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

ax2.set_title(f"SESUDAH PCA\n(Ruang Fitur 2 Dimensi | Total Retensi: {(var_pc1 + var_pc2):.2f}%)", fontweight='bold')
ax2.set_xlabel(f'Principal Component 1 ({var_pc1:.1f}%)')
ax2.set_ylabel(f'Principal Component 2 ({var_pc2:.1f}%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# Memunculkan layar
plt.tight_layout()
plt.show()

# --- TAMBAHKAN KODE INI DI BARIS PALING BAWAH ---
print("\n" + "="*50)
print("PARAMETER C++ UNTUK ESP32 (TAHAP EKSTRAKSI FITUR)")
print("="*50)

print("// 1. Parameter Standarisasi Sensor (X, Y, Z)")
print(f"const float mean_X = {scaler.mean_[0]:.6f};")
print(f"const float mean_Y = {scaler.mean_[1]:.6f};")
print(f"const float mean_Z = {scaler.mean_[2]:.6f};")
print(f"const float scale_X = {scaler.scale_[0]:.6f};")
print(f"const float scale_Y = {scaler.scale_[1]:.6f};")
print(f"const float scale_Z = {scaler.scale_[2]:.6f};\n")

print("// 2. Parameter Matriks PCA (Eigenvectors)")
print(f"const float pca_w1_x = {pca.components_[0][0]:.6f};")
print(f"const float pca_w1_y = {pca.components_[0][1]:.6f};")
print(f"const float pca_w1_z = {pca.components_[0][2]:.6f};")

print(f"const float pca_w2_x = {pca.components_[1][0]:.6f};")
print(f"const float pca_w2_y = {pca.components_[1][1]:.6f};")
print(f"const float pca_w2_z = {pca.components_[1][2]:.6f};\n")
print("="*50)