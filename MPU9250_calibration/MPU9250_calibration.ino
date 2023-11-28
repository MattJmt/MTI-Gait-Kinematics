#include <MPU9250_WE.h>
#include <Wire.h>
#define MPU9250_ADDR 0x69

MPU9250_WE myMPU9250 = MPU9250_WE(MPU9250_ADDR);
// define IMU addresses
const int IMU_1 = 5;
const int IMU_2 = 6;
const int IMU_3 = 7;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  myMPU9250.enableGyrDLPF();
  myMPU9250.setGyrDLPF(MPU9250_DLPF_6);
  myMPU9250.setGyrRange(MPU9250_GYRO_RANGE_250);
  myMPU9250.setAccRange(MPU9250_ACC_RANGE_2G);
  myMPU9250.enableAccDLPF(true);
  myMPU9250.setAccDLPF(MPU9250_DLPF_6);
  myMPU9250.enableAccAxes(MPU9250_ENABLE_XYZ);
  myMPU9250.enableGyrAxes(MPU9250_ENABLE_XYZ);
  myMPU9250.setMagOpMode(AK8963_CONT_MODE_100HZ);
  delay(200);
   pinMode(IMU_1, OUTPUT);
pinMode(IMU_2, OUTPUT);
pinMode(IMU_3, OUTPUT);
digitalWrite(IMU_1,LOW);
digitalWrite(IMU_2,LOW);
digitalWrite(IMU_3,LOW);
//  Serial.print("t, ");
//  Serial.print("ax1, ");Serial.print("ay1, ");Serial.print("az1, ");  //Serial.print("px1, ");Serial.print("py1, ");Serial.print("pz1, ");
//  Serial.print("ax2, ");Serial.print("ax2, ");Serial.print("az2, ");  //Serial.print("px1, ");Serial.print("py1, ");Serial.print("pz1, ");
//  Serial.print("ax3, ");Serial.print("ay3, ");Serial.print("az3, ");Serial.println();  //Serial.print("px1, ");Serial.print("py1, ");Serial.print("pz1, ");Serial.println();
}

int test[9] = {11,12,13,14,15,16,17,18,19};
float cal[3][12] = {
  {16365, 16397.5, 16695, 273.3, 183, 1118.3, 1, 1, 1},  // IMU 1 [gain x, gain y, gain z, offset x, offset y, offset z] x2
  {16409, 16380, 16595, 348.3, -40, 95, 1, 1, 1},        // IMU 2
  {16395, 16400, 16635, 294.5, -82.2, -119.7, 1, 1, 1}   // IMU 3
 };
//int gain_x_1 = 16365;
//int gain_y_1 = 16398;
//int gain_z_1 = 16695;
//int gain_x_2 = 16409;
//int gain_y_2 = 16380;
//int gain_z_2 = 16595;
//int gain_x_3 = 16395;
//int gain_y_3 = 16400;
//int gain_z_3 = 16635;
//int offset_x_1 = 273;
//int offset_y_1 = 183; 
//int offset_z_1 = 1118;
//int offset_x_2 = 348;
//int offset_y_2 = -40;
//int offset_z_2 = 95;
//int offset_x_3 = 295;
//int offset_y_3 = -82;
//int offset_z_3 = -120;
//int p_gain_x_1 = 1;
//int p_gain_y_1 = 1;
//int p_gain_z_1 = 1;
//int p_gain_x_2 = 1;
//int p_gain_y_2 = 1;
//int p_gain_z_2 = 1;
//int p_gain_x_3 = 1;
//int p_gain_y_3 = 1;
//int p_gain_z_3 = 1;
//int p_offset_x_1 = 0;
//int p_offset_y_1 = 0;
//int p_offset_z_1 = 0;
//int p_offset_x_2 = 0;
//int p_offset_y_2 = 0;
//int p_offset_z_2 = 0;
//int p_offset_x_3 = 0;
//int p_offset_y_3 = 0;
//int p_offset_z_3 = 0;

unsigned long previousMillis = 0;
void loop() {
  float accx_1, accy_1, accz_1, accx_2, accy_2, accz_2, accx_3, accy_3, accz_3, gyrx_1, gyry_1, gyrz_1, gyrx_2, gyry_2, gyrz_2, gyrx_3, gyry_3, gyrz_3;
  float t = ((float) millis())/1000.0; // time variable
  static float values[3][6];
  int j = 0;
  int pin = 5;
  unsigned long currentMillis = millis();
  xyzFloat accRaw; //= myMPU9250.getAccRawValues();
  xyzFloat gyrRaw; //= myMPU9250.getGyrRawValues();

//  xyzFloat gValue = myMPU9250.getGValues();
//  xyzFloat gyr = myMPU9250.getGyrValues();
//  xyzFloat magValue = myMPU9250.getMagValues();
//  float temp = myMPU9250.getTemperature();
//  float resultantG = myMPU9250.getResultantG(gValue);
  
//  if (currentMillis - previousMillis >= 10) {
//    // Save the current time for the next iteration
//    previousMillis = currentMillis;
//

 for (int pin = 5; pin <= 7; pin++) {
    
    digitalWrite( pin, HIGH);    // set this sensor at 0x69
    accRaw = myMPU9250.getAccRawValues();
    gyrRaw = myMPU9250.getGyrRawValues();

    values[j][0] = accRaw.x;
    values[j][1] = accRaw.y;
    values[j][2] = accRaw.z;

//    values[j][3] = gyrRaw.x;
//    values[j][4] = gyrRaw.y;
//    values[j][5] = gyrRaw.z;

    
    digitalWrite(pin, LOW);   // park this sensor at 0x68

    j ++ ;
  }
  j = 0;

  accx_1 = (values[0][0]-cal[0][3]) / cal[0][0];
  accy_1 = (values[0][1]-cal[0][4]) / cal[0][1];
  accz_1 = (values[0][2]-cal[0][5]) / cal[0][2];

  accx_2 = (values[1][0]-cal[1][3]) / cal[1][0];
  accy_2 = (values[1][1]-cal[1][4]) / cal[1][1];
  accz_2 = (values[1][2]-cal[1][5]) / cal[1][2];

  accx_3 = (values[2][0]-cal[2][3]) / cal[2][0];
  accy_3 = (values[2][1]-cal[2][4]) / cal[2][1];
  accz_3 = (values[2][2]-cal[2][5]) / cal[2][2];
//
//  gyrx_1 = (values[0][3]-cal[0][9]) / cal[0][6];
//  gyry_1 = (values[0][4]-cal[0][10]) / cal[0][7];
//  gyrz_1 = (values[0][5]-cal[0][11]) / cal[0][8];
//
//  gyrx_2 = (values[1][3]-cal[1][9]) / cal[1][6];
//  gyry_2 = (values[1][4]-cal[1][10]) / cal[1][7];
//  gyrz_2 = (values[1][5]-cal[1][11]) / cal[1][8];
//
//  gyrx_3 = (values[2][3]-cal[2][9]) / cal[2][6];
//  gyry_3 = (values[2][4]-cal[2][10]) / cal[2][7];
//  gyrz_3 = (values[2][5]-cal[2][11]) / cal[2][8];

  Serial.print(t);Serial.print(",");
  Serial.print(accx_1);Serial.print(",");Serial.print(accy_1);Serial.print(",");Serial.print(accz_1);Serial.print(","); //Serial.print(" ");Serial.print(gyrx_1);Serial.print(" ");Serial.print(gyry_1);Serial.print(" ");Serial.print(gyrz_1);Serial.print(" ");
  Serial.print(accx_2);Serial.print(",");Serial.print(accy_2);Serial.print(",");Serial.print(accz_2);Serial.print(","); //Serial.print(" ");Serial.print(gyrx_1);Serial.print(" ");Serial.print(gyry_1);Serial.print(" ");Serial.print(gyrz_1);Serial.print(" ");
  Serial.print(accx_3);Serial.print(",");Serial.print(accy_3);Serial.print(",");Serial.print(accz_3); //Serial.print(" ");Serial.print(gyrx_1);Serial.print(" ");Serial.print(gyry_1);Serial.print(" ");Serial.print(gyrz_1);Serial.print(" ");
  Serial.println();
  
//
//  Serial.println("Gyroscope data in degrees/s: ");
//  Serial.print(gyr.x);
//  Serial.print("   ");
//  Serial.print(gyr.y);
//  Serial.print("   ");
//  Serial.println(gyr.z);

//  Serial.println("Magnetometer Data in µTesla: ");
//  Serial.print(magValue.x);
//  Serial.print("   ");
//  Serial.print(magValue.y);
//  Serial.print("   ");
//  Serial.println(magValue.z);
//
//  Serial.print("Temperature in °C: ");
//  Serial.println(temp);


}
