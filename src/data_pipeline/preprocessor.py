from transformers import DistilBertTokenizer
import torch

class GameDataProcessor:
    def __init__(self, max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length
    
    def process(self, raw_data_path):
        # Carrega e converte dados brutos
        states_text = self._load_and_convert(raw_data_path)
        
        # Tokenização
        inputs = self.tokenizer(
            states_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Salva dados processados
        torch.save(inputs, 'data/processed/dataset.pt')
        return inputs