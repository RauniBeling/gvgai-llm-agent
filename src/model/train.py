import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from src.utils.metrics import compute_metrics

class GameTrainer:
    def __init__(self, model_path='distilbert-base-uncased'):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.training_args = TrainingArguments(
            output_dir='models/trained/',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='logs/',
            report_to="wandb"  # Integração com W&B
        )
    
    def train(self, train_dataset, eval_dataset):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model('models/trained/final_model')