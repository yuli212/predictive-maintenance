import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

try:
    df_normal = pd.read_excel("IMU_Normal.xlsx")
    df_kotor = pd.read_excel("IMU_Kotor.xlsx")
except FileNotFoundError:
    print("File Excel tidak ditemukan, cek nama file dan lokasinya")
    exit()

df_normal['Label'] = 0
df_kotor['Label'] = 1
df = pd.concat([df_normal, df_kotor], ignore_index=True)

kolom_fitur = ['Accel X (g)', 'Accel Y (g)', 'Accel Z (g)']
X = df[kolom_fitur]
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Label'] = y.values
df_pca['Kondisi'] = df_pca['Label'].map({0: 'Normal', 1: 'Indikasi Kotor'})

df_pca[['PC1', 'PC2', 'Kondisi']].to_excel("Dataset_Hasil_PCA.xlsx", index=False)
print(f"Hasil PCA disimpan ke Dataset_Hasil_PCA.xlsx")
print(f"Ukuran data: {X.shape} -> {df_pca.shape}")

# visualisasi sebelum dan sesudah PCA
fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X.loc[y==0, 'Accel X (g)'], X.loc[y==0, 'Accel Y (g)'], X.loc[y==0, 'Accel Z (g)'],
            c='blue', label='Normal', alpha=0.6, s=20)
ax1.scatter(X.loc[y==1, 'Accel X (g)'], X.loc[y==1, 'Accel Y (g)'], X.loc[y==1, 'Accel Z (g)'],
            c='red', label='Indikasi Kotor', alpha=0.6, s=20)
ax1.set_title("Sebelum PCA (3D)", fontweight='bold')
ax1.set_xlabel('Accel X (g)')
ax1.set_ylabel('Accel Y (g)')
ax1.set_zlabel('Accel Z (g)')
ax1.legend()

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

ax2 = fig.add_subplot(122)
ax2.scatter(df_pca.loc[df_pca['Label']==0, 'PC1'], df_pca.loc[df_pca['Label']==0, 'PC2'],
            c='blue', label='Normal', alpha=0.7, edgecolors='k', s=40)
ax2.scatter(df_pca.loc[df_pca['Label']==1, 'PC1'], df_pca.loc[df_pca['Label']==1, 'PC2'],
            c='red', label='Indikasi Kotor', alpha=0.7, edgecolors='k', s=40)
ax2.set_title(f"Sesudah PCA (2D) | Retensi: {(var_pc1+var_pc2):.2f}%", fontweight='bold')
ax2.set_xlabel(f'PC1 ({var_pc1:.1f}%)')
ax2.set_ylabel(f'PC2 ({var_pc2:.1f}%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()

# parameter untuk ESP32
print("\nParameter C++ untuk ekstraksi fitur di ESP32:")
print(f"const float mean_X = {scaler.mean_[0]:.6f};")
print(f"const float mean_Y = {scaler.mean_[1]:.6f};")
print(f"const float mean_Z = {scaler.mean_[2]:.6f};")
print(f"const float scale_X = {scaler.scale_[0]:.6f};")
print(f"const float scale_Y = {scaler.scale_[1]:.6f};")
print(f"const float scale_Z = {scaler.scale_[2]:.6f};")

print(f"\nEigenvector PCA:")
print(f"PC1: [{pca.components_[0][0]:.6f}, {pca.components_[0][1]:.6f}, {pca.components_[0][2]:.6f}]")
print(f"PC2: [{pca.components_[1][0]:.6f}, {pca.components_[1][1]:.6f}, {pca.components_[1][2]:.6f}]")
