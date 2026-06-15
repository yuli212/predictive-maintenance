import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from micromlgen import port

try:
    df = pd.read_excel("Dataset_Fitur_Vibrasi.xlsx")
except FileNotFoundError:
    print("File Dataset_Fitur_Vibrasi.xlsx tidak ditemukan, jalankan dummy_data.py dulu")
    exit()

X = df[['Varians_X', 'Varians_Z']].values
y = df['Kondisi'].values

print(f"Data dimuat: {len(df)} sampel ({(y==0).sum()} Normal, {(y==1).sum()} Kotor)\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Parameter scaler:")
print(f"  mean_var_x  = {scaler.mean_[0]:.6f}")
print(f"  mean_var_z  = {scaler.mean_[1]:.6f}")
print(f"  scale_var_x = {scaler.scale_[0]:.6f}")
print(f"  scale_var_z = {scaler.scale_[1]:.6f}\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# gamma=1.0 eksplisit agar micromlgen bisa ekstrak array C++ dengan benar
model_rbf = SVC(kernel='rbf', class_weight='balanced', C=50.0, gamma=1.0)
skor = cross_val_score(model_rbf, X_scaled, y, cv=kf, scoring='accuracy')
print(f"5-Fold CV: {[round(s*100, 2) for s in skor]}")
print(f"Rata-rata akurasi: {np.mean(skor)*100:.2f}%\n")

model_rbf.fit(X_scaled, y)
y_pred = model_rbf.predict(X_scaled)
print(classification_report(y, y_pred, target_names=['Normal', 'Indikasi Kotor']))
print(f"Jumlah support vectors: {len(model_rbf.support_vectors_)}")

output_path = "hardware/SvmRbfModel.h"
with open(output_path, 'w') as f:
    f.write(port(model_rbf, classmap={0: 'Normal', 1: 'Indikasi_Kotor'}))
print(f"\nModel diekspor ke {output_path}")

# visualisasi decision boundary
plt.figure(figsize=(9, 6))
plt.scatter(X_scaled[y==0][:, 0], X_scaled[y==0][:, 1],
            color='blue', label='Normal', edgecolors='k', alpha=0.7)
plt.scatter(X_scaled[y==1][:, 0], X_scaled[y==1][:, 1],
            color='red', label='Indikasi Kotor', edgecolors='k', alpha=0.7)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 200)
yy = np.linspace(ylim[0], ylim[1], 200)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_rbf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

plt.title('Decision Boundary SVM RBF (Fitur Varians)', fontweight='bold')
plt.xlabel('Varians X (standardized)')
plt.ylabel('Varians Z (standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
