# from envs.MPC_Nav_env import QuadrupedGymEnv
from envs.MPC_task import QuadrupedGymEnv
# from envs.MPC_task_high_speed import QuadrupedGymEnv
# from envs.MPC_task_accel_foothold import QuadrupedGymEnv
# from envs.ros.MPC_task_accel_foothold_ros import QuadrupedGymEnv
import pybullet as bc
import numpy as np
import time
import json
import csv

env = QuadrupedGymEnv(time_step=0.001, action_repeat=1, obs_hist_len=1,render = True)

env.reset()

data_log = []

for _ in range(20000):
    observation, reward, done, meas = env.step(action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0]))
    
    if _ % 10 == 0:  # Logs every 10th step
        data_log.append({
            "Time": time.time(),
            "Position": meas['base_pos'],
            "Velocity": meas['base_vel']
        })

    # if _ % 100 == 0:
    #     with open("simulation_data_mpc.json", "w") as json_file:
    #         json.dump(data_log, json_file)
    
    # env._settle_robot()
    # env.step(action = np.array([0.0, 0.0, 0.0]))
    # 0, 0, 0, 0, 0]))

# with open("simulation_data_mpc.json", "w") as json_file:
#     json.dump(data_log, json_file)

# Write data_log to CSV
with open("simulation_data_mpc.csv", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow(["Time", "Position_x", "Position_y", "Position_z", 
                         "Velocity_x", "Velocity_y", "Velocity_z"])

    # Write rows
    for entry in data_log:
        csv_writer.writerow([
            entry["Time"],
            entry["Position"][0], entry["Position"][1], entry["Position"][2],
            entry["Velocity"][0], entry["Velocity"][1], entry["Velocity"][2]
        ])
