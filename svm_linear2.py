import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from micromlgen import port

try:
    df = pd.read_excel("Dataset_Fitur_Vibrasi.xlsx")
except FileNotFoundError:
    print("File tidak ditemukan")
    exit()

X = df[['Varians_X', 'Varians_Z']].values
y = df['Kondisi'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_svm = SVC(kernel='linear', class_weight='balanced', C=1.0, gamma=1.0)
model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi: {akurasi*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Indikasi Kotor']))

# ekspor ke header arduino
c_header = port(model_svm, classmap={0: 'Normal', 1: 'Indikasi_Kotor'})
os.makedirs("hardware", exist_ok=True)
path_file_h = os.path.join("hardware", "SvmLinearModel.h")
with open(path_file_h, "w") as f:
    f.write(c_header)
print(f"Model disimpan ke: {path_file_h}")

print("\nParameter scaler untuk Arduino:")
print(f"const float mean_var_x = {scaler.mean_[0]:.6f};")
print(f"const float mean_var_z = {scaler.mean_[1]:.6f};")
print(f"const float scale_var_x = {scaler.scale_[0]:.6f};")
print(f"const float scale_var_z = {scaler.scale_[1]:.6f};")

# plot decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y==0][:, 0], X_scaled[y==0][:, 1],
            color='blue', label='Normal', edgecolors='k')
plt.scatter(X_scaled[y==1][:, 0], X_scaled[y==1][:, 1],
            color='red', label='Kotor (Gredek)', edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_svm.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

plt.title(f"Decision Boundary SVM Linear (Akurasi: {akurasi*100:.0f}%)", fontweight='bold')
plt.xlabel("Varians X (standardized)")
plt.ylabel("Varians Z (standardized)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
