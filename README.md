# MTI Gait Kinematics
Designing a wearable device to monitor gait for Multiple Sclerosis (MS) patients for my Medical Technology Innovation class. 
The device extracts step count and step period (cadence) from its user. It also models the gait kinematics using forward kinematics, as shown below.

<img src="https://github.com/MattJmt/MTI-Gait-Kinematics/blob/main/Pictures/WalkingTest.gif" width="170"/> <img src="https://github.com/MattJmt/MTI-Gait-Kinematics/blob/main/figures/Walking_animation.gif" width="425"/>





The [MTI-Report](https://github.com/MattJmt/MTI-Gait-Kinematics/blob/main/MTI-Report.pdf)  gives more context to MS and the relevance of such a wearable device to monitor its symptoms.

The IMU accelerations are read and calibrated in [MPU9250_Calibration.ino](https://github.com/MattJmt/MTI-Gait-Kinematics/blob/main/MPU9250_calibration/MPU9250_calibration.ino) and are then filtered using the ```scipy.signal``` *butter()* and *sosfiltfilt()* functions in ```Serial.py```, after which step count and period, along with a 3D kinematic animation, are displayed in ```Model.py```. 
![Unfilt and Filt accelerations](https://github.com/MattJmt/MTI-Gait-Kinematics/assets/80914835/6d27883d-429d-45c2-8436-1d108b7f9d3d)

The step count and period are extracted from the knee joint, since it has the cleanest and most reliable signal. This has also been proven in literature. 

![KneeJoint_Filtered_Acc](https://github.com/MattJmt/MTI-Gait-Kinematics/assets/80914835/392c6173-0acb-42e3-bfa4-9199f9c3c3f6)

Future improvements include:
  - transitioning from the i2c cable communication to wireless bluetooth
  - adding ml to detect more detailed gait metrics for different users (eg. step height, knee/hip angular extension,...), and correlating them to heart rate (ECG), muscle activity (EMG), and other physiological measurements as well as professional health practitioners diagnoses.
