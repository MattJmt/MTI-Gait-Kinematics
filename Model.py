import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy import signal

filename = 'data3.csv'
df = pd.read_csv(filename)
t = df['Time'].values

# Function to convert DataFrame columns to a NumPy array
def df_to_np(df, cols):
    return df[cols].values

# Acceleration arrays
a1 = df_to_np(df, ['Acc1_X', 'Acc1_Y', 'Acc1_Z'])
a2 = df_to_np(df, ['Acc2_X', 'Acc2_Y', 'Acc2_Z'])
a3 = df_to_np(df, ['Acc3_X', 'Acc3_Y', 'Acc3_Z'])

# Filtered Acceleration arrays
a1_f = df_to_np(df, ['AccF1_X', 'AccF1_Y', 'AccF1_Z'])
a2_f = df_to_np(df, ['AccF2_X', 'AccF2_Y', 'AccF2_Z'])
a3_f = df_to_np(df, ['AccF3_X', 'AccF3_Y', 'AccF3_Z'])

# Velocity arrays
v1 = df_to_np(df, ['Vel1_X', 'Vel1_Y', 'Vel1_Z'])
v2 = df_to_np(df, ['Vel2_X', 'Vel2_Y', 'Vel2_Z'])
v3 = df_to_np(df, ['Vel3_X', 'Vel3_Y', 'Vel3_Z'])

# Position arrays
p1 = df_to_np(df, ['Pos1_X', 'Pos1_Y', 'Pos1_Z'])
p2 = df_to_np(df, ['Pos2_X', 'Pos2_Y', 'Pos2_Z'])
p3 = df_to_np(df, ['Pos3_X', 'Pos3_Y', 'Pos3_Z'])

initial_pos1 = np.array([0.2,0,0.0])  # placed 0.2m from ground, but assume foot, 0.2m from center
initial_pos2 = np.array([0.2,0,0.55]) # placed 0.6m from ground

""" 
Data Processing
"""

## Plot Accelerations
imu_acc = [a1[:,0],a1[:,1],a1[:,2],a2[:,0],a2[:,1],a2[:,2],a3[:,0],a3[:,1],a3[:,2],
           a1_f[:,0],a1_f[:,1],a1_f[:,2],a2_f[:,0],a2_f[:,1],a2_f[:,2],a3_f[:,0],a3_f[:,1],a3_f[:,2]]
imu_acc_label = ['ax1','ay1','az1','ax2','ay2','az2','ax3','ay3','az3',
                 'ax1f','ay1f','az1f','ax2f','ay2f','az2f','ax3f','ay3f','az3f']

for i in range(12,len(imu_acc)-3):
    plt.plot(t,imu_acc[i], label = imu_acc_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('IMU Accelerations')
plt.legend(bbox_to_anchor = (1.1, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Velocities
imu_vel = [v1[:,0],v1[:,1],v1[:,2],v2[:,0],v2[:,1],v2[:,2],v3[:,0],v3[:,1],v3[:,2]]
imu_vel_label = ['vx1','vy1','vz1','vx2','vy2','vz2','vx3','vy3','vz3']
#t = t[:-1]          # remove 1 time index due to integration

for i in range(3,len(imu_vel)-3):
    plt.plot(t,imu_vel[i], label = imu_vel_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('IMU Velocities')
plt.legend(bbox_to_anchor = (1.1, 0.6), loc='center right')
plt.grid(True)
plt.show()

## Plot Positions
imu_pos = [p1[:,0],p1[:,1],p1[:,2],p2[:,0],p2[:,1],p2[:,2],p3[:,0],p3[:,1],p3[:,2]]
imu_pos_label = ['px1','py1','pz1','px2','py2','pz2','px3','py3','pz3']
#t = t[:-1]          # remove 1 time index due to integration
for i in range(3,len(imu_pos)-3):
    plt.plot(t,imu_pos[i], label = imu_pos_label[i], linewidth = 1)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('IMU Positions')
plt.legend(bbox_to_anchor = (1.1, 0.6), loc='center right')
plt.grid(True)
plt.show()

## EXTRACT NUMBER OF STEPS AND CADENCE
ax2_peaks, _ = signal.find_peaks(a2_f[:,0], height=0.1)
ay2_peaks, _ = signal.find_peaks(-a2_f[:,1], height=0.4)
az2_peaks, _ = signal.find_peaks(a2_f[:,2], height=-0.9)
print(
      "ax2_peaks:", len(ax2_peaks),
      "ay1_peaks:", len(ay2_peaks),
      "az2_peaks:", len(az2_peaks)
      )

steps = min(len(ay2_peaks), len(az2_peaks))
sum = 0
for i in range (0,len(ay2_peaks)-1):
    sum += t[ay2_peaks[i+1]] - t[ay2_peaks[i]]
    
step_period = sum / (len(ay2_peaks)-1)
print("steps:{} | step period (s): {:.2f}".format(steps, step_period))


""" 
3D ANIMATION
"""

p1[:,1] = 2*p1[:,1] + p2[:,1]

## initiate joints for left leg, half a period later
time_index = []
time_i = []

# find period of a step, when the knee joint moves backwards, assumin start at straight/standing position
for i in range(0,len(p2)):
    if p2[i,1] < 0 and p2[i-1,1] > 0:
        time_index += [[t[i],i]]
# print("time index", time_index)

for i in range(0,len(time_index)):
    if time_index[i][0] > 2:
        time_i = time_index[i]
        break
# print(time_i)

# gait_start_i = time_i[1]
gait_start_i = 15



# create array with initial position 1 for duration of leg to complete half a period
no_moving1 = np.full((gait_start_i,3),initial_pos1)         
no_moving2 = np.full((gait_start_i,3),initial_pos2)

# join "waiting" array with start of right leg array, and remove end of right leg array corresponding to length of "waiting" array
p1_left = np.vstack((no_moving1, p1[:-gait_start_i]))        
p2_left = np.vstack((no_moving2, p2[:-gait_start_i]))

## create fixed join p4 (head)
p4 = np.full((len(p1),3),np.array([0,0,1.9]))       # 1.9m height

time_list = t
x_lists = [p1[:,0], p2[:,0], p3[:,0], -p2_left[:,0], -p1_left[:,0], p4[:,0]]     # -x for symmetry of left leg
y_lists = [p1[:,1], p2[:,1], p3[:,1], p2_left[:,1], p1_left[:,1], p4[:,1]]
z_lists = [p1[:,2], p2[:,2], p3[:,2], p2_left[:,2], p1_left[:,2], p4[:,2]]

# Number of joints
num_joints = len(x_lists)
print(num_joints)

# Create a figure and axis for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])  # Adjust the x-axis limits as needed
ax.set_ylim([-1.5, 1.5])  # Adjust the y-axis limits as needed
ax.set_zlim([0, 2])  # Adjust the z-axis limits as needed
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Animation of Joints')

# Initialize empty lines for the joints
lines = [ax.plot([], [], [], marker='o')[0] for _ in range(num_joints-1)]

time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Function to update the plot for each frame of the animation
def update(frame):
    for i in range(num_joints):

        if i == 5:  # Connecting p6 to p4
            lines[-1].set_data([x_lists[i][frame], x_lists[2][frame]], [y_lists[i][frame], y_lists[2][frame]])
            lines[-1].set_3d_properties([z_lists[i][frame], z_lists[2][frame]])

        elif i > 0 and i != 5:
            # Update lines to connect the joints
            lines[i - 1].set_data([x_lists[i - 1][frame], x_lists[i][frame]], [y_lists[i - 1][frame], y_lists[i][frame]])
            lines[i - 1].set_3d_properties([z_lists[i - 1][frame], z_lists[i][frame]])

    time_text.set_text('Time: {:.2f} seconds'.format(time_list[frame]) + " | steps:{} | step period (s): {:.2f}".format(steps, step_period))
    return lines + [time_text]

# Function to update the view angles for rotation (azimuth and elevation)
def update_view(event):
    ax.view_init(elev=elevation_slider.val, azim=azimuth_slider.val)

ax.view_init(elev=20, azim=5)

# Create sliders for interactive control of elevation and azimuth
elevation_slider_ax = plt.axes([0.1, 0.02, 0.65, 0.03])
azimuth_slider_ax = plt.axes([0.1, 0.06, 0.65, 0.03])
elevation_slider = plt.Slider(elevation_slider_ax, 'Elevation', -90, 90, valinit=20)
azimuth_slider = plt.Slider(azimuth_slider_ax, 'Azimuth', -180, 180, valinit=25)

# Update view when sliders are changed
elevation_slider.on_changed(update_view)
azimuth_slider.on_changed(update_view)

#Create the animation using FuncAnimation
frames_per_second = 10  # For example, 10 frames per second
duration_of_animation = t[-1]  # Duration of the animation in seconds
frames = int(frames_per_second * duration_of_animation)
frames = min(len(p1), len(p2), len(p3))  # Number of frames (assumes equal lengths for all joints)
ani = animation.FuncAnimation(fig, update, frames=frames, interval = 1000/frames_per_second, blit=True)
writer1 = animation.PillowWriter(fps=30)  # Create a PillowWriter instance
ani.save('Walking_animation0.gif', writer=writer1)
plt.show()
