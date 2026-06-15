import numpy as np
import pandas as pd
import os

print("Membentuk Data Simulasi IMU V3 (Skenario Jagrak Tengah & Pre-Filtered)...\n")

n_rows = 200
waktu = np.linspace(0, 4, n_rows)

# --- 1. GENERATE DATA NORMAL (IDEAL & FILTERED) ---
def generate_imu_normal_filtered():
    # Murni getaran natural mesin (misal 5Hz). 
    # Karena sudah difilter dan motor di jagrak, tidak ada noise acak (random.normal)
    base_engine = np.sin(2 * np.pi * 5 * waktu) * 0.05
    
    # Sumbu X dan Y dianggap nyaris diam karena getaran linear mesin rapi di sumbu Z
    accel_x = np.zeros(n_rows) 
    accel_y = np.zeros(n_rows)
    # Sumbu Z dipengaruhi gravitasi (1g) + getaran mesin
    accel_z = 1.0 + base_engine
    
    return pd.DataFrame({'Accel X (g)': accel_x, 'Accel Y (g)': accel_y, 'Accel Z (g)': accel_z}).round(3)

# --- 2. GENERATE DATA KOTOR (IDEAL & FILTERED) ---
def generate_imu_kotor_filtered():
    base_engine = np.sin(2 * np.pi * 5 * waktu) * 0.05
    
    # Getaran CVT kotor (frekuensi lebih tinggi, misal 25Hz) yang menembus batas filter
    # Amplitudonya besar karena gesekan mekanis yang nyata
    getaran_cvt = np.sin(2 * np.pi * 25 * waktu) * 0.25 
    
    # Karena getarannya kasar, getaran merambat sedikit ke sumbu horizontal (Sumbu X)
    accel_x = np.sin(2 * np.pi * 25 * waktu) * 0.08
    accel_y = np.zeros(n_rows)
    accel_z = 1.0 + base_engine + getaran_cvt
    
    # Spikes mekanis (lonjakan fisis nyata akibat mangkok ganda selip)
    spike_idx = np.random.choice(n_rows, 5, replace=False)
    accel_z[spike_idx] += np.random.uniform(0.5, 0.8, 5)
    
    return pd.DataFrame({'Accel X (g)': accel_x, 'Accel Y (g)': accel_y, 'Accel Z (g)': accel_z}).round(3)

# --- 3. EKSEKUSI ---
df_normal = generate_imu_normal_filtered()
df_kotor = generate_imu_kotor_filtered()

df_normal.to_excel("IMU_Normal.xlsx", index=False)
df_kotor.to_excel("IMU_Kotor.xlsx", index=False)

print("[SUKSES] Data telah diperbarui sesuai batasan: Motor di Jagrak & Data Ter-filter!")