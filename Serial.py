import serial
import time
import numpy as np
import matplotlib.pyplot as plt



# Replace 'COM3' with the appropriate port name for your system (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
ser = serial.Serial('COM4', 9600, timeout=1)  # Open serial port with baud rate 9600
# data_saved_string = np.array([])
# data_saved = np.array([])
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
                
            # Add a small delay to not overload the serial port
            #time.sleep(0.1)

except KeyboardInterrupt:
    print("Keyboard Interrupt. Closing serial port.")
    ser.close()  # Close the serial port
except serial.SerialException as e:
    print("Serial Exception:", e)
    ser.close()  # Close the serial port if an exception occurs

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

## Accelerations
t = output_array[:,0]           # time
a1 = output_array[:,1:4]        # IMU 1 acc vector
a2 = output_array[:,4:7]        # IMU 2 acc vector
a3 = output_array[:,7:10]       # IMU 3 acc vector

## rotation matrix
# convert raw xyz accelerations to intertial frame. 

def compute_rot_matrix(a1):
    vector1 = a1[0]        # extract initial acc vector
    print("raw vector ",vector1)
    desired_frame = np.array([0,0,-1])                # g force only in z direction
    print("des vector ", desired_frame)

    vector1 = vector1 / np.linalg.norm(vector1)
    desired_frame = desired_frame / np.linalg.norm(desired_frame)

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
    print("rot mat ", rotation_matrix)
    
    ## Convert all vectors to desired frame
    for i in range(0,len(a1)):
        vector_i = a1[i] 
        vectori_des = np.dot(rotation_matrix, vector_i)
        a1[i] = vectori_des
    return a1, rotation_matrix

a1_des, rot_mat1 = compute_rot_matrix(a1)
a2_des, rot_mat2 = compute_rot_matrix(a2)
a3_des, rot_mat3 = compute_rot_matrix(a3)

check1 = np.dot(a1[0],rot_mat1)
check2 = np.dot(a2[0],rot_mat2)
check3 = np.dot(a3[0],rot_mat3)
print("check", check1,check2,check3)
print(a1_des[0],a2_des[0],a3_des[0])

imu_acc = [a1[:,0],a1[:,1],a1[:,2],a2[:,0],a2[:,1],a2[:,2],a3[:,0],a3[:,1],a3[:,2]]
imu_acc_label = ['ax1','ay1','az1','ax2','ay2','az2','ax3','ay3','az3']

for i in range(0,len(imu_acc)):
    plt.plot(t,imu_acc[i], label = imu_acc_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('IMU Accelerations')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

def euler(accelerations, timestamps):
    velocities = [0.0]  # Initial velocity at timestamp 0
    positions = [0.0]    # Initial position at timestamp 0
    time_diff = np.diff(timestamps)  # Calculate time differences

    for i in range(1,len(accelerations)):
        # Assuming constant acceleration between irregular timestamps
        delta_t = time_diff[i] if i < len(time_diff) else time_diff[-1]  # Last time diff for extrapolation

        # Calculate velocity using Euler's method
        #new_velocity = velocities[-1] + (accelerations[i]/accelerations[0]) * delta_t
        new_velocity = (accelerations[i] - accelerations[i-1]) * delta_t * 9.81
        velocities.append(new_velocity)

        # Calculate position using velocity and Euler's method


        new_position = positions[-1] + velocities[-1] * delta_t * 9.81
        positions.append(new_position)

    # Convert velocity and position lists to NumPy arrays
    velocities_array = np.array(velocities)
    positions_array = np.array(positions)
    velocities_array = velocities_array[:len(velocities_array)]
    positions_array = positions_array[:len(positions_array)]

    return velocities_array, positions_array

## Velocities
vx1, px1 = euler(ax1,t)     # IMU 1
vy1, py1 = euler(ay1,t)
vz1, pz1 = euler(az1,t)
vx2, px2 = euler(ax2,t)
vy2, py2 = euler(ay2,t)
vz2, pz2 = euler(az2,t)     
vx3, px3 = euler(ax3,t)
vy3, py3 = euler(ay3,t)
vz3, pz3 = euler(az3,t)

#from initial position
initial_pos1 = np.array([0,0.2,0.0])  # placed 0.2m from ground, but assume foot, 0.2m from center
initial_pos2 = np.array([0,0.2,0.55]) # placed 0.6m from ground
initial_pos3 = np.array([0,0,1.05]) # placed 1.1m from ground
px1 = initial_pos1[0] - px1
py1 = initial_pos1[1] - py1
pz1 = initial_pos1[2] - pz1
px2 = initial_pos2[0] - px2
py2 = initial_pos2[1] - py2
pz2 = initial_pos2[2] - pz2
px3 = initial_pos3[0] - px3
py3 = initial_pos3[1] - py3
pz3 = initial_pos3[2] - pz3


## Plot Accelerations
imu_acc = [ax1,ay1,az1,ax2,ay2,az2,ax3,ay3,az3]
imu_acc_label = ['ax1','ay1','az1','ax2','ay2','az2','ax3','ay3','az3']

for i in range(0,len(imu_acc)):
    plt.plot(t,imu_acc[i], label = imu_acc_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('IMU Accelerations')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Velocities
imu_vel = [vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]
imu_vel_label = ['vx1','vy1','vz1','vx2','vy2','vz2','vx3','vy3','vz3']

# plot of all IMU readings
for i in range(0,len(imu_vel)):
    plt.plot(t,imu_vel[i], label = imu_vel_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('IMU Velocities')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Positions
imu_pos = [px1,py1,pz1,px2,py2,pz2,px3,py3,pz3]
imu_pos_label = ['px1','py1','pz1','px2','py2','pz2','px3','py3','pz3']

# plot of all IMU readings
for i in range(0,len(imu_pos)):
    plt.plot(t,imu_pos[i], label = imu_pos_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Position m')
plt.title('IMU Positions')
plt.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
plt.grid(True)
plt.show()