import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from .base_model import BaseModel

class TextDataset(Dataset):
    """Custom Dataset for BERT text processing"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx]) if hasattr(self.texts, 'iloc') else str(self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    """BERT-based classification model"""
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.linear(output)

class BERTModel(BaseModel):
    """
    BERT model for text classification with automatic GPU/CPU handling
    """

    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 max_length=128,
                 batch_size=16,
                 learning_rate=2e-5,
                 epochs=3,
                 dropout=0.1):
        super().__init__("BERT")
        self.bert_model_name = bert_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def build_model(self):
        """Build the BERT model architecture"""
        self.model = BertClassifier(
            bert_model_name=self.bert_model_name,
            num_classes=2,
            dropout=self.dropout
        )
        self.model.to(self.device)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the BERT model

        Args:
            X_train: Training features (text data)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            float: Training time in seconds
        """
        if self.model is None:
            self.build_model()

        # Create datasets and dataloaders
        train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Setup optimizer and scheduler - using torch.optim.AdamW instead of transformers.AdamW
        # Setup optimizer and scheduler - using torch.optim.AdamW instead of transformers.AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Loss function
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        # Training loop
        start_time = time.time()
        self.model.train()

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)

            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f'Average training loss: {avg_loss:.4f}')

            # Optional validation
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")

        self.training_time = time.time() - start_time
        return self.training_time

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data

        Args:
            X_test: Test features (text data)
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        test_dataset = TextDataset(X_test, y_test, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())

        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(actual_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(actual_labels, predictions, average='weighted', zero_division=0)

        # Get classification report and confusion matrix
        report = classification_report(actual_labels, predictions, target_names=['0:Correct', '1:Error'])
        cm = confusion_matrix(actual_labels, predictions)

        self.accuracy = accuracy

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def predict(self, X):
        """
        Predict class labels for samples in X

        Args:
            X: Features (text data)

        Returns:
            array: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        dataset = TextDataset(X, [0]*len(X), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X

        Args:
            X: Features (text data)

        Returns:
            array: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        dataset = TextDataset(X, [0]*len(X), self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)

                probabilities.extend(probs.cpu().tolist())

        return np.array(probabilities)

    def compare_predictions(self, X, y):
        """
        Compare true labels with predicted labels

        Args:
            X: Features (text data)
            y: True labels

        Returns:
            DataFrame: Comparison of true and predicted labels
        """
        y_pred = self.predict(X)
        comparison = pd.DataFrame({
            "True Label": y,
            "Predicted Label": y_pred
        })
        return comparison

    def save_model_here(self):
        """
        Save the trained model and tokenizer in the current working directory.
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving.")

        # Get current working directory
        save_path = os.getcwd()
        folder_path = os.path.join(save_path, "bert_model")
        os.makedirs(folder_path, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(folder_path, "bert_classifier.pt"))

        # Save tokenizer
        self.tokenizer.save_pretrained(folder_path)

        print(f"Model and tokenizer saved to {folder_path}")
