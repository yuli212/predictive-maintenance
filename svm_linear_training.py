import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from micromlgen import port

try:
    df = pd.read_excel("Dataset_Fitur_Vibrasi.xlsx")
except FileNotFoundError:
    print("File tidak ditemukan, jalankan dummy_data.py dulu")
    exit()

X = df[['Varians_X', 'Varians_Z']].values
y = df['Kondisi'].values

print(f"Data dimuat: {len(df)} sampel ({(y==0).sum()} Normal, {(y==1).sum()} Kotor)\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# print parameter scaler buat disalin ke arduino
print("Parameter scaler untuk hardware.ino:")
print(f"  mean_var_x  = {scaler.mean_[0]:.6f}")
print(f"  mean_var_z  = {scaler.mean_[1]:.6f}")
print(f"  scale_var_x = {scaler.scale_[0]:.6f}")
print(f"  scale_var_z = {scaler.scale_[1]:.6f}\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = SVC(kernel='linear', class_weight='balanced', gamma=1.0)
skor = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
print(f"5-Fold CV: {[round(s*100, 2) for s in skor]}")
print(f"Rata-rata akurasi: {np.mean(skor)*100:.2f}%\n")

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
print(classification_report(y, y_pred, target_names=['Normal', 'Indikasi Kotor']))

output_path = "hardware/SvmLinearModel.h"
with open(output_path, 'w') as f:
    f.write(port(model, classmap={0: 'Normal', 1: 'Indikasi_Kotor'}))

print(f"Model diekspor ke {output_path}")
