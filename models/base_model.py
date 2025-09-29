from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

class BaseModel(ABC):
    """
    Abstract base class for all models in the comparison.
    """

    def __init__(self, name):
        """
        Initialize the base model.

        Args:
            name (str): Name of the model
        """
        self.name = name
        self.model = None
        self.training_time = None
        self.accuracy = None

    @abstractmethod
    def build_model(self):
        """Build the model architecture. To be implemented by subclasses."""
        pass

    def train(self, X_train, y_train):
        """
        Train the model and measure training time.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            float: Training time in seconds
        """
        if self.model is None:
            self.build_model()

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        return self.training_time

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred, target_names=['0:Correct', '1:Error'])
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': self.accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def get_summary(self):
        """Get a summary of the model's performance."""
        return {
            'name': self.name,
            'training_time': self.training_time,
            'accuracy': self.accuracy
        }