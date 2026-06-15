import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_excel("Dataset_Hasil_PCA.xlsx")
except FileNotFoundError:
    print("File Dataset_Hasil_PCA.xlsx tidak ditemukan")
    exit()

X = df[['PC1', 'PC2']].values
y = df['Kondisi'].map({'Normal': 0, 'Indikasi Kotor': 1}).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_svm = SVC(kernel='rbf', class_weight='balanced', C=50.0, gamma='scale')

skor_akurasi = cross_val_score(model_svm, X_scaled, y, cv=kf, scoring='accuracy')
print(f"Akurasi per fold: {[round(acc*100, 2) for acc in skor_akurasi]}")
print(f"Rata-rata: {np.mean(skor_akurasi)*100:.2f}%\n")

model_svm.fit(X_scaled, y)
y_pred = model_svm.predict(X_scaled)
print(classification_report(y, y_pred, target_names=['Normal', 'Indikasi Kotor']))

print(f"Jumlah support vectors: {len(model_svm.support_vectors_)}")

# visualisasi hyperplane
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[y==0][:, 0], X_scaled[y==0][:, 1],
            color='blue', label='Normal', edgecolors='k', alpha=0.7)
plt.scatter(X_scaled[y==1][:, 0], X_scaled[y==1][:, 1],
            color='red', label='Indikasi Kotor', edgecolors='k', alpha=0.7)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_svm.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

plt.title('Batas Keputusan SVM RBF', fontweight='bold')
plt.xlabel('PC1 (Standardized)')
plt.ylabel('PC2 (Standardized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
