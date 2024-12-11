import pandas as pd
import matplotlib.pyplot as plt

# Load both CSV files
data_mpc = pd.read_csv("simulation_data_mpc.csv")
data_mlp1 = pd.read_csv("simulation_data_mlp1.csv")
data_mlp = pd.read_csv("simulation_data_mlp.csv")

# Adjust the time series for both files
times_mpc = data_mpc["Time"] - (data_mpc["Time"].iloc[0])
times_mlp = data_mlp["Time"] - (data_mlp["Time"].iloc[0])

# Find the maximum time to plot, based on the shorter time series
max_time = min(times_mpc.iloc[-1], times_mlp.iloc[-1])

# Extract position components for both datasets
position_x_mpc = data_mpc["Position_x"]
position_y_mpc = data_mpc["Position_y"]
position_z_mpc = data_mpc["Position_z"]
velocity_x_mpc = data_mpc["Velocity_x"]
velocity_y_mpc = data_mpc["Velocity_y"]
velocity_z_mpc = data_mpc["Velocity_z"]

position_x_mlp1 = data_mlp1["Position_x"]
position_y_mlp1 = data_mlp1["Position_y"]
position_z_mlp1 = data_mlp1["Position_z"]
velocity_x_mlp1 = data_mlp1["Velocity_x"]
velocity_y_mlp1 = data_mlp1["Velocity_y"]
velocity_z_mlp1 = data_mlp1["Velocity_z"]

position_x_mlp = data_mlp["Position_x"]
position_y_mlp = data_mlp["Position_y"]
position_z_mlp = data_mlp["Position_z"]
velocity_x_mlp = data_mlp["Velocity_x"]
velocity_y_mlp = data_mlp["Velocity_y"]
velocity_z_mlp = data_mlp["Velocity_z"]

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot positions on the first subplot
axes[0].plot(times_mpc, position_x_mpc, 'b--', label="px_mpc")  # Blue dashed for MPC x
axes[0].plot(times_mlp, position_x_mlp, 'b-', label="px_mlp")   # Blue solid for MLP x
axes[0].plot(times_mpc, position_y_mpc, 'g--', label="py_mpc")  # Green dashed for MPC y
axes[0].plot(times_mlp, position_y_mlp, 'g-', label="py_mlp")   # Green solid for MLP y
axes[0].plot(times_mpc, position_z_mpc, 'r--', label="pz_mpc")  # Red dashed for MPC z
axes[0].plot(times_mlp, position_z_mlp, 'r-', label="pz_mlp")   # Red solid for MLP z
axes[0].set_xlim([0, max_time])
axes[0].set_ylabel("Position [m]")
axes[0].set_title("Robot Body Position")
axes[0].legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# Plot velocities on the second subplot
axes[1].plot(times_mpc, velocity_x_mpc, 'b--', label="vx_mpc")  # Blue dashed for MPC x velocity
axes[1].plot(times_mlp, velocity_x_mlp, 'b-', label="vx_mlp")   # Blue solid for MLP x velocity
axes[1].plot(times_mpc, velocity_y_mpc, 'g--', label="vy_mpc")  # Green dashed for MPC y velocity
axes[1].plot(times_mlp, velocity_y_mlp, 'g-', label="vy_mlp")   # Green solid for MLP y velocity
axes[1].plot(times_mpc, velocity_z_mpc, 'r--', label="vz_mpc")  # Red dashed for MPC z velocity
axes[1].plot(times_mlp, velocity_z_mlp, 'r-', label="vz_mlp")   # Red solid for MLP z velocity
axes[1].set_xlim([0, max_time])
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity [m/s]")
axes[1].set_title("Robot Body Velocity")
axes[1].legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# # Plot positions on the first subplot (1 horizon comparison)
# axes[0].plot(times_mlp, position_x_mlp1, 'b--', label="px 1 horizon obs")  # Blue dashed for MPC x
# axes[0].plot(times_mlp, position_x_mlp, 'b-', label="px 5 horizon obs")   # Blue solid for MLP x
# axes[0].plot(times_mlp, position_y_mlp1, 'g--', label="py 1 horizon obs")  # Green dashed for MPC y
# axes[0].plot(times_mlp, position_y_mlp, 'g-', label="py 5 horizon obs")   # Green solid for MLP y
# axes[0].plot(times_mlp, position_z_mlp1, 'r--', label="pz 1 horizon obs")  # Red dashed for MPC z
# axes[0].plot(times_mlp, position_z_mlp, 'r-', label="pz 5 horizon obs")   # Red solid for MLP z
# axes[0].set_xlim([0, max_time])
# axes[0].set_ylabel("Position [m]")
# axes[0].set_title("Robot Body Position")
# axes[0].legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# # Plot velocities on the second subplot (1 horizon comparison)
# axes[1].plot(times_mlp, velocity_x_mlp1, 'b--', label="vx 1 horizon obs")  # Blue dashed for MPC x velocity
# axes[1].plot(times_mlp, velocity_x_mlp, 'b-', label="vx 5 horizon obs")   # Blue solid for MLP x velocity
# axes[1].plot(times_mlp, velocity_y_mlp1, 'g--', label="vy 1 horizon obs")  # Green dashed for MPC y velocity
# axes[1].plot(times_mlp, velocity_y_mlp, 'g-', label="vy 5 horizon obs")   # Green solid for MLP y velocity
# axes[1].plot(times_mlp, velocity_z_mlp1, 'r--', label="vz 1 horizon obs")  # Red dashed for MPC z velocity
# axes[1].plot(times_mlp, velocity_z_mlp, 'r-', label="vz 5 horizon obs")   # Red solid for MLP z velocity
# axes[1].set_xlim([0, max_time])
# axes[1].set_xlabel("Time [s]")
# axes[1].set_ylabel("Velocity [m/s]")
# axes[1].set_title("Robot Body Velocity")
# axes[1].legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# Adjust layout for better spacing
plt.tight_layout()

# Show the combined plot
plt.show()


# # Plot positions over adjusted time for both logs with consistent colors
# plt.figure()
# plt.plot(times_mpc, position_x_mpc, 'b--', label="px_mpc")  # Blue dashed for MPC x
# plt.plot(times_mlp, position_x_mlp, 'b-', label="px_mlp")   # Blue solid for MLP x
# plt.plot(times_mpc, position_y_mpc, 'g--', label="py_mpc")  # Green dashed for MPC y
# plt.plot(times_mlp, position_y_mlp, 'g-', label="py_mlp")   # Green solid for MLP y
# plt.plot(times_mpc, position_z_mpc, 'r--', label="pz_mpc")  # Red dashed for MPC z
# plt.plot(times_mlp, position_z_mlp, 'r-', label="pz_mlp")   # Red solid for MLP z

# # Set x-axis limit to end at the minimum of the two time series lengths
# plt.xlim([0, max_time])

# # Adjust legend to the right side, outside the plot
# plt.legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# plt.xlabel("Time [s]")
# plt.ylabel("Position [m]")
# plt.title("Robot Body Position")

# # Show plot
# plt.show()

# # Plot velocities over adjusted time for both logs with consistent colors
# plt.figure()
# plt.plot(times_mpc, velocity_x_mpc, 'b--', label="vx_mpc")  # Blue dashed for MPC x velocity
# plt.plot(times_mlp, velocity_x_mlp, 'b-', label="vx_mlp")   # Blue solid for MLP x velocity
# plt.plot(times_mpc, velocity_y_mpc, 'g--', label="vy_mpc")  # Green dashed for MPC y velocity
# plt.plot(times_mlp, velocity_y_mlp, 'g-', label="vy_mlp")   # Green solid for MLP y velocity
# plt.plot(times_mpc, velocity_z_mpc, 'r--', label="vz_mpc")  # Red dashed for MPC z velocity
# plt.plot(times_mlp, velocity_z_mlp, 'r-', label="vz_mlp")   # Red solid for MLP z velocity

# # Set x-axis limit to end at the minimum of the two time series lengths
# plt.xlim([0, max_time])

# # Adjust legend to the right side, outside the plot
# plt.legend(loc="upper right", bbox_to_anchor=(1.10, 1), borderaxespad=0.)

# plt.xlabel("Time [s]")
# plt.ylabel("Velocity [m/s]")
# plt.title("Robot Body Velocity")

# # Show velocity plot
# plt.show()
