from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import time
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from .base_model import BaseModel

class BartMNLIModel(BaseModel):
    """
    BART-large-mNLI model for zero-shot classification and contradiction detection.
    """

    def __init__(self, model_name="facebook/bart-large-mnli", max_length=512, device=None):
        """
        Initialize the BART-mNLI model.

        Args:
            model_name (str): Hugging Face model name
            max_length (int): Maximum sequence length
            device (str): Device to run on ('cuda' or 'cpu')
        """
        super().__init__("BART-large-mNLI")
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.model = None
        self.tokenizer = None

    def build_model(self):
        """Build the BART-mNLI model and tokenizer."""
        try:
            # Option 1: Using pipeline for simplicity
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == 'cuda' else -1
            )

            # Option 2: Direct model loading for more control
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)

        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def train(self, X_train, y_train):
        """
        Note: BART-mNLI is a pre-trained model that doesn't require training.
        This method is implemented for interface compatibility.

        Args:
            X_train: Training features (unused for zero-shot model)
            y_train: Training labels (unused for zero-shot model)

        Returns:
            float: Training time (0 for zero-shot model)
        """
        if self.model is None:
            self.build_model()

        # Zero-shot models don't require training
        self.training_time = 0.0
        return self.training_time

    def predict(self, texts, candidate_labels, multi_label=False):
        """
        Predict labels for given texts using zero-shot classification.

        Args:
            texts (str or list): Text(s) to classify
            candidate_labels (list): Possible labels
            multi_label (bool): Whether multiple labels can be true

        Returns:
            dict or list: Prediction results
        """
        if self.pipeline is None:
            self.build_model()

        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            result = self.pipeline(
                text,
                candidate_labels,
                multi_label=multi_label,
                hypothesis_template="This example is about {}."
            )
            results.append(result)

        return results[0] if len(results) == 1 else results

    def detect_contradiction(self, premise, hypothesis):
        """
        Detect contradiction between two statements.

        Args:
            premise (str): First statement
            hypothesis (str): Second statement to check against premise

        Returns:
            dict: Contradiction analysis results
        """
        if self.model is None or self.tokenizer is None:
            self.build_model()

        # Tokenize the premise-hypothesis pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)

        # BART-mNLI output indices: 0=contradiction, 1=neutral, 2=entailment
        contradiction_prob = probs[0][0].item()
        neutral_prob = probs[0][1].item()
        entailment_prob = probs[0][2].item()

        # Determine the relationship
        max_prob = max(contradiction_prob, neutral_prob, entailment_prob)
        if max_prob == contradiction_prob:
            relationship = "contradiction"
        elif max_prob == entailment_prob:
            relationship = "entailment"
        else:
            relationship = "neutral"

        return {
            'relationship': relationship,
            'contradiction_prob': contradiction_prob,
            'neutral_prob': neutral_prob,
            'entailment_prob': entailment_prob,
            'premise': premise,
            'hypothesis': hypothesis
        }

    def evaluate(self, X_test, y_test, candidate_labels):
        """
        Evaluate the model on test data for zero-shot classification.

        Args:
            X_test: Test texts
            y_test: True labels (must be within candidate_labels)
            candidate_labels (list): Possible classification labels

        Returns:
            dict: Evaluation metrics
        """
        if self.pipeline is None:
            self.build_model()

        y_pred = []
        y_true = []

        # Convert X_test to list if it's a single string
        if isinstance(X_test, str):
            X_test = [X_test]
            y_test = [y_test]

        # Make predictions
        for text, true_label in zip(X_test, y_test):
            result = self.predict(text, candidate_labels)
            predicted_label = result['labels'][0]  # Take the highest probability label
            y_pred.append(predicted_label)
            y_true.append(true_label)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        report = classification_report(y_true, y_pred, target_names=candidate_labels)
        cm = confusion_matrix(y_true, y_pred, labels=candidate_labels)

        self.accuracy = accuracy

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': list(zip(X_test, y_true, y_pred))
        }

    def evaluate_contradictions(self, test_pairs):
        y_true = []
        y_pred = []

        for premise, hypothesis, true_relationship in test_pairs:
            result = self.detect_contradiction(premise, hypothesis)
            predicted_relationship = result['relationship']

            y_true.append(true_relationship)
            y_pred.append(predicted_relationship)

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'detailed_results': list(zip([p[0] for p in test_pairs], [p[1] for p in test_pairs], y_true, y_pred))
        }


    def compare_predictions(self, X, y_true, candidate_labels):
        """
        Compare true vs predicted labels.

        Args:
            X: Input texts
            y_true: True labels
            candidate_labels: Possible labels

        Returns:
            DataFrame: Comparison of predictions
        """
        import pandas as pd

        if isinstance(X, str):
            X = [X]
            y_true = [y_true]

        comparisons = []
        for text, true_label in zip(X, y_true):
            result = self.predict(text, candidate_labels)
            pred_label = result['labels'][0]
            pred_score = result['scores'][0]

            comparisons.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'prediction_score': pred_score,
                'is_correct': true_label == pred_label
            })

        return pd.DataFrame(comparisons)