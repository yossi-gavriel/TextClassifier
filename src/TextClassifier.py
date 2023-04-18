import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from TextDataset import TextDataset

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, train_texts, train_labels):
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(3):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                label = batch['label'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=label)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, texts):
        test_dataset = TextDataset(texts, [0]*len(texts), self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions += probs.tolist()

        return predictions