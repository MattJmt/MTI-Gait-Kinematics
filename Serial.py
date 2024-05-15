import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal

# Define CSV filename
filename = "data3.csv"

# Replace 'COM4' with the appropriate port name for your system (e.g., '/dev/ttyUSB0' on Linux or 'COM4' on Windows)
ser = serial.Serial('COM4', 9600, timeout=1)  # Open serial port with baud rate 9600
data_saved = np.array([])
data_saved_string = []
try:
    if ser.is_open:
        print("Serial port is open.")

        while True:
            # Read data from serial port
            data = ser.readline().decode().strip()
            if data:
                print("Received:", data)
                data_saved_string += [data.split(",")]

except KeyboardInterrupt:
    print("Keyboard Interrupt. Closing serial port.")
    ser.close()  # Close the serial port
except serial.SerialException as e:
    print("Serial Exception:", e)
    ser.close()  # Close the serial port if an exception occurs

# convert the strings from the serial port to floats
def convert_strings_to_floats(input_array):
    output_array = np.array([])
    array_list = []
    for i in range(0,len(input_array)):
        row = np.array([])
        for element in input_array[i]:
            converted_float = float(element)
            row = np.append(row,[converted_float])
            
        array_list += [row]
        output_array = np.vstack(array_list)
    return output_array

output_array = convert_strings_to_floats(data_saved_string)
output_array = output_array[1:]

# Extract raw accelerations
t = output_array[:,0]           # time
a1 = output_array[:,1:4]        # IMU 1 acc vector
a2 = output_array[:,4:7]        # IMU 2 acc vector
a3 = output_array[:,7:10]       # IMU 3 acc vector

# convert raw xyz accelerations to intertial frame. 
def compute_rot_matrix(a1):
    vector1 = a1[0]        # extract initial acc vector
    desired_frame = np.array([0,0,-1])                # g force only in z direction
    
    # normalise the vectors
    vector1 /= np.linalg.norm(vector1)
    desired_frame /= np.linalg.norm(desired_frame)

     # Calculate rotation axis using cross product - 
    rotation_axis = np.cross(vector1, desired_frame)

    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize rotation axis

    # Calculate rotation angle
    dot_product = np.dot(vector1, desired_frame)
    rotation_angle = np.arccos(dot_product)

    # Construct rotation matrix using Rodrigues' formula
    skew_symmetric = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                               [rotation_axis[2], 0, -rotation_axis[0]],
                               [-rotation_axis[1], rotation_axis[0], 0]])
    
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * skew_symmetric + \
                      (1 - np.cos(rotation_angle)) * np.dot(skew_symmetric, skew_symmetric)
    
    ## Convert all vectors to desired frame
    for i in range(0,len(a1)):
        vector_i = a1[i] 
        vectori_des = np.dot(rotation_matrix, vector_i)
        a1[i] = vectori_des
    return a1, rotation_matrix

a1_des, rot_mat1 = compute_rot_matrix(a1)
a2_des, rot_mat2 = compute_rot_matrix(a2)
a3_des, rot_mat3 = compute_rot_matrix(a3)

# Check whether rotation matrix succesful
# check1 = np.dot(a1[0],rot_mat1)
# check2 = np.dot(a2[0],rot_mat2)
# check3 = np.dot(a3[0],rot_mat3)
# print("check", check1,check2,check3)
# print(a1_des[0],a2_des[0],a3_des[0])

def euler(accelerations, timestamps):
    initial_vel = np.array([0.0, 0.0, 0.0])  # Initial velocity at timestamp 0
    initial_pos = np.array([0.0, 0.0, 0.0])    # Initial position at timestamp 0
    time_diff = np.diff(timestamps)  # Calculate time differences

    vel_array = [initial_vel]
    pos_array = [initial_pos]
    for i in range(1,len(accelerations)):
        # Assuming constant acceleration between irregular timestamps
        delta_t = time_diff[i] if i < len(time_diff) else time_diff[-1]  # Last time diff for extrapolation

        # Calculate velocity using Euler's method
        new_velocity = (accelerations[i]-accelerations[i-1]) * delta_t * 9.81
        vel_array += [new_velocity]
        velocities = np.vstack(vel_array)

        # Calculate position using velocity and Euler's method
        new_pos = pos_array[-1] + velocities[-1] * delta_t * 9.81
        pos_array += [new_pos]
        positions = np.vstack(pos_array)

    return velocities, positions

# filter the raw accelerations in forward and backward direction with Butterworth 
def filtfilt(acceleration,CO):
    
    sos = signal.butter(3, CO, 'lp', fs=50, output='sos')
    acceleration_x_filt = signal.sosfiltfilt(sos, acceleration[:,0])
    acceleration_y_filt = signal.sosfiltfilt(sos, acceleration[:,1])
    acceleration_z_filt = signal.sosfiltfilt(sos, acceleration[:,2])

    acceleration_filt = np.column_stack((acceleration_x_filt, acceleration_y_filt, acceleration_z_filt))
    return acceleration_filt

a1_f = filtfilt(a1,5)
a2_f = filtfilt(a2,5)
a3_f = filtfilt(a3,5)

# Calculate velocity and position vectors using euler approximation
v1, p1 = euler(a1_f,t)     
v2, p2 = euler(a2_f,t)    
v3, p3 = euler(a3_f,t)

v1 = filtfilt(v1,15)
v2 = filtfilt(v2,15)
v3 = filtfilt(v3,15)

# add initial position of sensors on user
initial_pos1 = np.array([0.2,0,0.0])  # placed 0.2m from ground, but assume foot, 0.2m from center
initial_pos2 = np.array([0.2,0,0.55]) # placed 0.6m from ground
initial_pos3 = np.array([0,0,1.05]) # placed 1.1m from ground
p1 = initial_pos1 - p1
p2 = initial_pos2 - p2
p3 = initial_pos3 - p3

## Plot Accelerations
imu_acc = [a1[:,0],a1[:,1],a1[:,2],a2[:,0],a2[:,1],a2[:,2],a3[:,0],a3[:,1],a3[:,2],
           a1_f[:,0],a1_f[:,1],a1_f[:,2],a2_f[:,0],a2_f[:,1],a2_f[:,2],a3_f[:,0],a3_f[:,1],a3_f[:,2]]
imu_acc_label = ['ax1','ay1','az1','ax2','ay2','az2','ax3','ay3','az3',
                 'ax1f','ay1f','az1f','ax2f','ay2f','az2f','ax3f','ay3f','az3f']

for i in range(0,len(imu_acc)):
    plt.plot(t,imu_acc[i], label = imu_acc_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('IMU Accelerations')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Velocities
imu_vel = [v1[:,0],v1[:,1],v1[:,2],v2[:,0],v2[:,1],v2[:,2],v3[:,0],v3[:,1],v3[:,2]]
imu_vel_label = ['vx1','vy1','vz1','vx2','vy2','vz2','vx3','vy3','vz3']

for i in range(0,len(imu_vel)):
    plt.plot(t,imu_vel[i], label = imu_vel_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('IMU Velocities')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Positions
imu_pos = [p1[:,0],p1[:,1],p1[:,2],p2[:,0],p2[:,1],p2[:,2],p3[:,0],p3[:,1],p3[:,2]]
imu_pos_label = ['px1','py1','pz1','px2','py2','pz2','px3','py3','pz3']

for i in range(0,len(imu_pos)):
    plt.plot(t,imu_pos[i], label = imu_pos_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Position m')
plt.title('IMU Positions')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

# Save the data to a CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Writing the headers (if you know what each column represents, replace these generic headers)
    writer.writerow(['Time', 'Acc1_X', 'Acc1_Y', 'Acc1_Z', 'Acc2_X', 'Acc2_Y', 'Acc2_Z', 'Acc3_X', 'Acc3_Y', 'Acc3_Z',
                     'AccF1_X', 'AccF1_Y', 'AccF1_Z', 'AccF2_X', 'AccF2_Y', 'AccF2_Z', 'AccF3_X', 'AccF3_Y', 'AccF3_Z',
                     'Vel1_X', 'Vel1_Y', 'Vel1_Z', 'Vel2_X', 'Vel2_Y', 'Vel2_Z', 'Vel3_X', 'Vel3_Y', 'Vel3_Z',
                     'Pos1_X', 'Pos1_Y', 'Pos1_Z', 'Pos2_X', 'Pos2_Y', 'Pos2_Z', 'Pos3_X', 'Pos3_Y', 'Pos3_Z'])
    # Write the data
    for i in range(len(t)):
        row = [t[i]] + \
              a1[i].tolist() + a2[i].tolist() + a3[i].tolist() + \
              a1_f[i].tolist() + a2_f[i].tolist() + a3_f[i].tolist() + \
              v1[i].tolist() + v2[i].tolist() + v3[i].tolist() + \
              p1[i].tolist() + p2[i].tolist() + p3[i].tolist()
        writer.writerow(row)