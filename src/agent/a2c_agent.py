import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

class A2CAgent:
    def __init__(self, env_name='GVGAI-sokoban-v0'):
        self.env = make_vec_env(env_name, n_envs=4)
        self.model = A2C(
            "MlpPolicy", 
            self.env,
            learning_rate=7e-4,
            verbose=1,
            device='cuda'  # Habilite se tiver GPU
        )
    
    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)
    
    def save(self, path):
        self.model.save(path)