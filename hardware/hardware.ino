#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "SvmLinearModel.h"

Adafruit_MPU6050 mpu;
Eloquent::ML::Port::SVM clf;

// nilai ini didapat dari output svm_linear_training.py
const float mean_var_x  = 0.020705f;
const float mean_var_z  = 0.079038f;
const float scale_var_x = 0.020042f;
const float scale_var_z = 0.071892f;

void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("\n========================================");
  Serial.println("MONITORING KONDISI CVT MOTOR");
  Serial.println("========================================");

  Wire.begin();

  if (!mpu.begin()) {
    Serial.println("MPU6050 tidak terdeteksi, cek kabel!");
    while (1) delay(10);
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("Sensor OK, monitoring dimulai...\n");
}

void loop() {
  float sum_gx = 0, sum_gz = 0;
  float sum_gx2 = 0, sum_gz2 = 0;

  // ambil 100 sampel dalam 1 detik
  for (int i = 0; i < 100; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    float gx = a.acceleration.x / 9.81f;
    float gz = a.acceleration.z / 9.81f;

    sum_gx  += gx;
    sum_gz  += gz;
    sum_gx2 += gx * gx;
    sum_gz2 += gz * gz;

    delay(10);
  }

  // hitung varians dari window 1 detik
  float mgx = sum_gx / 100.0f;
  float mgz = sum_gz / 100.0f;
  float var_x = (sum_gx2 / 100.0f) - (mgx * mgx);
  float var_z = (sum_gz2 / 100.0f) - (mgz * mgz);

  // normalisasi sebelum masuk model
  float input_x = (var_x - mean_var_x) / scale_var_x;
  float input_z = (var_z - mean_var_z) / scale_var_z;

  float fitur[2] = {input_x, input_z};
  int hasil = clf.predict(fitur);

  Serial.print("Var_X: "); Serial.print(var_x, 6);
  Serial.print("  Var_Z: "); Serial.print(var_z, 6);
  Serial.print("  -> ");
  Serial.println(hasil == 1 ? "Indikasi_Kotor" : "Normal");
}
