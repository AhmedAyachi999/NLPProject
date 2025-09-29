import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest model with TF-IDF for text features.
    """

    def __init__(self, n_estimators=150, max_features=500, class_weight='balanced', random_state=42):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators (int): Number of trees in the forest
            max_features (int): Maximum number of features for TF-IDF
            class_weight (str or dict): Weight associated with classes
            random_state (int): Random state for reproducibility
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def build_model(self):
        self.model = Pipeline(steps=[
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

    def train(self, X, y):
        """
        Train the model.

        Args:
            X: Features (should be a pandas Series or DataFrame with Kurzcode column)
            y: Labels

        Returns:
            float: Training time in seconds
        """
        # Extract the Kurzcode text data
        text_data = X

        # Build model if not already built
        if not hasattr(self, 'model') or self.model is None:
            self.build_model()

        # Train the model
        import time
        start_time = time.time()
        self.model.fit(text_data, y)
        return time.time() - start_time

    def evaluate(self, X, y):
        """
        Evaluate the model.

        Args:
            X: Features (should be a pandas Series or DataFrame with Kurzcode column)
            y: Labels

        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.model.predict(X)
        print(X.head(20))
        # Make predictions
        accuracy = self.model.score(X, y)

        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        return {'accuracy': accuracy,
                'recall':recall,
                'precision':precision}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def compare_predictions(self, X, y):
        y_pred = self.model.predict(X)
        comparison = pd.DataFrame({
            "True Label": y,
            "Predicted Label": y_pred
        })
        return comparison