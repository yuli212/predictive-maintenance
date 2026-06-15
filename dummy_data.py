import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 200

# data normal - getaran halus, nilai kecil
var_x_normal = np.abs(np.random.normal(loc=0.002, scale=0.001, size=n_samples))
var_z_normal = np.abs(np.random.normal(loc=0.010, scale=0.003, size=n_samples))

df_normal = pd.DataFrame({
    'Varians_X': var_x_normal,
    'Varians_Z': var_z_normal,
    'Kondisi': 0,
    'Label': 'Normal'
})

# kondisi kotor - getaran lebih besar dan fluktuatif
var_x_kotor = np.abs(np.random.normal(loc=0.040, scale=0.010, size=n_samples))
var_z_kotor = np.abs(np.random.normal(loc=0.150, scale=0.030, size=n_samples))

df_kotor = pd.DataFrame({
    'Varians_X': var_x_kotor,
    'Varians_Z': var_z_kotor,
    'Kondisi': 1,
    'Label': 'Indikasi Kotor'
})

df = pd.concat([df_normal, df_kotor], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_excel("Dataset_Fitur_Vibrasi.xlsx", index=False)

print(f"Dataset berhasil dibuat: {len(df)} baris")

# visualisasi sebaran data
plt.figure(figsize=(9, 6))
plt.scatter(df_normal['Varians_X'], df_normal['Varians_Z'],
            c='blue', label='Normal', alpha=0.7, edgecolors='k')
plt.scatter(df_kotor['Varians_X'], df_kotor['Varians_Z'],
            c='red', label='Indikasi Kotor (Gredek)', alpha=0.7, edgecolors='k')

plt.title("Sebaran Fitur Varians Getaran\n(tiap titik = 1 detik observasi)", fontweight='bold')
plt.xlabel("Varians Sumbu X (g²)")
plt.ylabel("Varians Sumbu Z (g²)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
