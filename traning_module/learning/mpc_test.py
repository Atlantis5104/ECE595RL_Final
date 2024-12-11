# from envs.MPC_Nav_env import QuadrupedGymEnv
from envs.MPC_task import QuadrupedGymEnv
# from envs.MPC_task_high_speed import QuadrupedGymEnv
# from envs.MPC_task_accel_foothold import QuadrupedGymEnv
# from envs.ros.MPC_task_accel_foothold_ros import QuadrupedGymEnv
import pybullet as bc
import numpy as np
import time
env = QuadrupedGymEnv(time_step=0.001, action_repeat=1, obs_hist_len=1)
env.reset()

for _ in range(20000):
    env.step(action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # env._settle_robot()
    # env.step(action = np.array([0.0, 0.0, 0.0]))
    # 0, 0, 0, 0, 0]))
