import pytest
from src.agent.a2c_agent import A2CAgent

@pytest.fixture
def agent():
    return A2CAgent()

def test_agent_initialization(agent):
    assert agent.env is not None
    assert agent.model is not None

def test_agent_training(agent):
    agent.train(timesteps=1000)
    assert True  # Verifica se o treinamento não falha