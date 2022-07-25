import gym
from HydroThermalEnv_1d import HydroThermalEnv, DEMAND

# Small example
env = HydroThermalEnv()
env.reset()
for i in range(99):
    action = DEMAND
    #action = 0
    env.step(action)

env.cumulative_reward
