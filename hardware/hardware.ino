#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "SvmRbfModel.h"

Adafruit_MPU6050 mpu;
Eloquent::ML::Port::SVM clf;

const float mean_pc1 = -0.000000;
const float mean_pc2 = -0.000000;
const float scale_pc1 = 1.361476;
const float scale_pc2 = 0.382601;

// Buffer Penyimpanan Data Sensor (Tetap dipertahankan untuk rolling di latar belakang)
float buffer_accel_x[100];
float buffer_accel_y[100];
float buffer_accel_z[100];
int buffer_index = 0;

void setup() {
  Serial.begin(115200);

  // Jeda 3 detik agar laptop sempat membuka koneksi USB Serial Monitor
  delay(3000);

  Serial.println("\n========================================");
  Serial.println("MEMULAI PENGUJIAN SERIAL & MPU6050");
  Serial.println("========================================");

  Wire.begin();

  // Mengecek respon I2C MPU6050
  if (!mpu.begin()) {
    Serial.println("[ERROR] Gagal menemukan MPU6050!");
    Serial.println("Periksa kabel 3V3, GND, D22 (SCL), dan D21 (SDA).");
    while (1) {
      delay(10); 
    }
  }

  Serial.println("[SUKSES] MPU6050 Ditemukan.");
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("[INFO] Sistem Siap! Mulai membaca akselerasi Sumbu Z...\n");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Rolling buffer latar belakang
  buffer_accel_x[buffer_index] = a.acceleration.x;
  buffer_accel_y[buffer_index] = a.acceleration.y;
  buffer_accel_z[buffer_index] = a.acceleration.z;
  buffer_index = (buffer_index + 1) % 100;

  // Mencetak nilai getaran Sumbu Z ke Serial Monitor (Mode Siaga)
  Serial.print("Getaran Z: ");
  Serial.println(a.acceleration.z);

  // Deteksi motor di-gas (Nilai Z melonjak melebihi bias gravitasi 12.0)
  if (a.acceleration.z > 12.0) {
    
    Serial.println("\n>>> TRIGGER GAS TERDETEKSI! Merekam getaran aktual (1 detik)...");

    // A) MEREKAM 100 SAMPEL GETARAN AKTUAL (Real-time data guncangan)
    float avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
    
    for (int i = 0; i < 100; i++) {
      sensors_event_t evt_a, evt_g, evt_t;
      mpu.getEvent(&evt_a, &evt_g, &evt_t);

      avg_x += evt_a.acceleration.x;
      avg_y += evt_a.acceleration.y;
      avg_z += evt_a.acceleration.z;

      // Jeda 10ms untuk mendapatkan sampling rate 100Hz
      delay(10); 
    }
    
    avg_x /= 100.0;
    avg_y /= 100.0;
    avg_z /= 100.0;

    Serial.println(">>> MENGHITUNG AI SVM RBF...");

    // B) Standarisasi Sensor (Menghilangkan bias gravitasi)
    float std_x = (avg_x - 0.0) / 0.039896;
    float std_y = (avg_y - 0.0) / 1.0;
    float std_z = (avg_z - 1.007783) / 0.150307;

    // C) Proyeksi PCA (Mengubah 3 Dimensi ke 2 Dimensi)
    float raw_pc1 = (std_x * 0.707107) + (std_y * 0.0) + (std_z * 0.707107);
    float raw_pc2 = (std_x * 0.707107) + (std_y * 0.0) + (std_z * -0.707107);

    // D) Standarisasi Model
    float scaled_pc1 = (raw_pc1 - 0.0) / 1.361476;
    float scaled_pc2 = (raw_pc2 - 0.0) / 0.382601;

    // E) Eksekusi Prediksi
    float model_input[2] = {scaled_pc1, scaled_pc2};
    int hasil_prediksi = clf.predict(model_input);

    if (hasil_prediksi == 1) {
      Serial.println(">>> KESIMPULAN: KOTOR (GREDEK)\n");
    } else {
      Serial.println(">>> KESIMPULAN: NORMAL\n");
    }

    Serial.println(">>> Menunggu 3 detik sebelum siaga kembali...\n");
    delay(3000); 
  }

  // Polling mode siaga setiap 100ms
  delay(100);
}