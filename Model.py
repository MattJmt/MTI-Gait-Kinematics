import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Serial import t, p1,p2,p3, initial_pos1, initial_pos2
import math

## initiate joints for left leg, a half a period later
time_index = []
time_i = []
for i in range(0,len(p2)):
    if p2[i,1] < 0 and p2[i-1,1] > 0:
        time_index += [[t[i],i]]
print("time index", time_index)

for i in range(0,len(time_index)):
    if time_index[i][0] > 2:
        time_i = time_index[i]
        break
print(time_i)
gait_start_i = time_i[1]

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

    time_text.set_text('Time: {:.2f} seconds'.format(time_list[frame]))
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
ani.save('Walking_animation8.gif', writer=writer1)
plt.show()
