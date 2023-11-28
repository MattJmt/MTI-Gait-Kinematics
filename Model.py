import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Serial import t, px1, px2, px3, py1, py2, py3, pz1, pz2, pz3

# initiate other legs
px1_left = -px1
py1_left = -py1
pz1_left =  pz1
px2_left = -px2
py2_left = -py2
pz2_left = pz2

# Function to generate joint positions using externally provided lists
def generate_joint_positions(x_lists, y_lists, z_lists, joint_index, frame):
    x_positions = x_lists[joint_index]
    y_positions = y_lists[joint_index]
    z_positions = z_lists[joint_index]
    
    x = x_positions[frame]
    y = y_positions[frame]
    z = z_positions[frame]
    
    return x, y, z

time_list = t
x_lists = [px1, px2, px3, px2_left, px1_left]
y_lists = [py1, py2, py3, py2_left, py1_left]
z_lists = [pz1, pz2, pz3, pz2_left, pz1_left]

# Number of joints
num_joints = len(x_lists)

# Create a figure and axis for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])  # Adjust the x-axis limits as needed
ax.set_ylim([-1.5, 1.5])  # Adjust the y-axis limits as needed
ax.set_zlim([0, 1.2])  # Adjust the z-axis limits as needed
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Animation of Joints')

# # Initialize empty lines for the joints
# lines = [ax.plot([], [], [], marker='o', markersize=5)[0] for _ in range(num_joints)]
# Initialize empty lines for the joints
lines = [ax.plot([], [], [], marker='o')[0] for _ in range(num_joints-1)]

time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Function to update the plot for each frame of the animation
def update(frame):
    for i in range(num_joints):
        x, y, z = generate_joint_positions(x_lists, y_lists, z_lists, i, frame)
        # lines[i].set_data([x], [y])
        # lines[i].set_3d_properties([z])
        if i > 0:
            # Update lines to connect the joints
            lines[i - 1].set_data([x_lists[i - 1][frame], x_lists[i][frame]], [y_lists[i - 1][frame], y_lists[i][frame]])
            lines[i - 1].set_3d_properties([z_lists[i - 1][frame], z_lists[i][frame]])
    time_text.set_text('Time: {:.2f} seconds'.format(time_list[frame]))
    return lines + [time_text]

# Function to update the view angles for rotation (azimuth and elevation)
def update_view(event):
    ax.view_init(elev=elevation_slider.val, azim=azimuth_slider.val)

# Create sliders for interactive control of elevation and azimuth
elevation_slider_ax = plt.axes([0.1, 0.02, 0.65, 0.03])
azimuth_slider_ax = plt.axes([0.1, 0.06, 0.65, 0.03])
elevation_slider = plt.Slider(elevation_slider_ax, 'Elevation', -90, 90, valinit=30)
azimuth_slider = plt.Slider(azimuth_slider_ax, 'Azimuth', -180, 180, valinit=-30)

# Update view when sliders are changed
elevation_slider.on_changed(update_view)
azimuth_slider.on_changed(update_view)

#Create the animation using FuncAnimation
frames_per_second = 10  # For example, 10 frames per second
duration_of_animation = t[-1]  # Duration of the animation in seconds
frames = int(frames_per_second * duration_of_animation)
frames = min(len(px1), len(px2), len(px3))  # Number of frames (assumes equal lengths for all joints)
animation = FuncAnimation(fig, update, frames=frames, interval = 1000/frames_per_second, blit=True)

# Display the animation
plt.show()
