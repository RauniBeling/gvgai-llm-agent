import json  
from datetime import datetime  
import gym  

class DataCollector:  
    def __init__(self, game_name='GVGAI-sokoban-v0'):  
        self.env = gym.make(game_name)  
        self.data = []  

    def collect(self, agent, episodes=100):  
        for episode in range(episodes):  
            state = self.env.reset()  
            done = False  
            while not done:  
                action = agent.predict(state)  
                next_state, reward, done, _ = self.env.step(action)  
                self.data.append({  
                    "state": state.tolist(),  
                    "action": int(action),  
                    "reward": float(reward),  
                    "timestamp": datetime.now().isoformat()  
                })  
                state = next_state  
            self._save_episode(episode)  

    def _save_episode(self, episode):  
        with open(f'data/raw/episode_{episode}.json', 'w') as f:  
            json.dump(self.data, f, indent=4)  