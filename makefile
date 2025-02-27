install:
    pip install -r requirements.txt

train:
    python -m src.agent.a2c_agent --timesteps 10000

process:
    python -m src.data_pipeline.preprocessor

train-model:
    python -m src.model.train

test:
    pytest tests/

clean:
    rm -rf __pycache__ .pytest_cache