from src.agent.a2c_agent import A2CAgent
from src.data_pipeline.collector import DataCollector
from src.data_pipeline.preprocessor import GameDataProcessor
from src.model.train import GameTrainer

def main():
    # Treinamento do agente base
    agent = A2CAgent()
    agent.train(timesteps=10000)

    # Coleta de dados
    collector = DataCollector()
    collector.collect(agent, episodes=50)

    # Pré-processamento
    processor = GameDataProcessor()
    processed_data = processor.process('data/raw/episode_0.json')

    # Treinamento do modelo
    trainer = GameTrainer()
    trainer.train(processed_data['train'], processed_data['eval'])

if __name__ == "__main__":
    main()