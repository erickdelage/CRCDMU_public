import gym
from gym import spaces
import numpy as np

alpha = 0.9
C_THERMAL = -3
V0 = 40
V_MAX = 100
DEMAND = 30
INFLOW = 20
rho = -0.3

class HydroThermalEnv(gym.Env):
    """A power system inventory environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(HydroThermalEnv, self).__init__()

        self.reward_range = (DEMAND * C_THERMAL / (1-alpha),0)

        # State space of the format stored energy 0 <= v <= V_MAX
        self.observation_space = spaces.Box(
            low=0, high=V_MAX, shape = (1,), dtype=np.float32)

        # Actions space of the format turbined energy 0 <= q <= DEMAND
        self.action_space = spaces.Box(
            low=0, high=DEMAND, shape = (1,), dtype=np.float32)

    def _next_observation(self):
        # Get the next stored energy state
        self.stored = min(
                self.stored + INFLOW - self.turbined, V_MAX)
        return self.stored

    def _take_action(self, action):
        turbined = min(action, self.stored + INFLOW)
        self.reward = rho * abs(action - turbined)
        self.turbined = turbined

    def step(self, action):
        self._take_action(action)

        thermal_reward = C_THERMAL * (DEMAND - self.turbined)
        self.reward += thermal_reward
        self.cumulative_reward += (alpha ** self.stage) * self.reward
        self.cumulative_thermal_reward += (alpha ** self.stage) * thermal_reward
        self.stage += 1

        done = False

        return self._next_observation(), self.reward, done, {}

    def reset(self):
        self.stored = V0
        self.turbined = 0
        self.stage = 0
        self.cumulative_reward = 0
        self.cumulative_thermal_reward = 0

        return self.stored

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Stored energy: {self.stored}')
        print(f'Cumulative reward: {self.cumulative_reward}')
        print(f'Cumulative thermal reward: {self.cumulative_thermal_reward}')
