import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from data_preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.BertModel import BERTModel

class ModelComparator:
    """
    Compares different models on the dataset.
    """

    def __init__(self, data_file):
        self.data_file = data_file
        self.models = []
        self.predictions = {}

    def add_model(self, model):
        self.models.append(model)

    def run_comparison(self):
        preprocessor = DataPreprocessor(self.data_file)
        X, y = preprocessor.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        for model in self.models:
            print(f"\n=== Testing {model.name} ===")
            try:
                model.train(X_train, y_train)
                evaluation = model.evaluate(X_test, y_test)
                y_pred = model.predict(X_test)

                self.predictions[model.name] = {
                    'y_true': y_test,
                    'y_pred': y_pred
                }

                print(f"Accuracy: {evaluation['accuracy']:.4f}")
                print(f"Precision: {evaluation.get('precision', 0):.4f}")
                print(f"Recall: {evaluation.get('recall', 0):.4f}")
                print(f"F1: {evaluation.get('f1', 0):.4f}")
            except Exception as e:
                print(f"Error testing {model.name}: {e}")

    def visualize_predictions(self):
        if not self.predictions:
            print("No predictions to visualize. Run the comparison first.")
            return

        for model_name, pred_data in self.predictions.items():
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']

            plt.figure(figsize=(6, 5))
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()

def main():
    comparator = ModelComparator('synth_errs.csv')
    #comparator.add_model(RandomForestModel())
    comparator.add_model(BERTModel())
    comparator.run_comparison()
    comparator.visualize_predictions()

    # Save BERT model after training
    for model in comparator.models:
        if isinstance(model, BERTModel):
            model.save_model_here()  # <-- saves in current directory

    print("\n=== Manual Prediction Mode ===")
    print("Enter row data for prediction (type 'quit' to exit)")

    while True:
        print("\n" + "="*50)
        avo_kurztext = input("AVO-Kurztext: ")
        if avo_kurztext.lower() == 'quit':
            break

        avo_langtext = input("AVO-Langtext: ")
        vd_kurztext = input("VD-Kurztext: ")
        beschreibung = input("Beschreibung: ")
        kurzcode = input("Kurzcode: ")

        manual_text = (
                avo_kurztext + " " +
                avo_langtext + " " +
                vd_kurztext + " " +
                beschreibung + " " +
                kurzcode
        )

        X_manual = [manual_text]

        print("\nPrediction Results:")
        for model in comparator.models:
            prediction = model.predict(X_manual)[0]
            probability = model.predict_proba(X_manual)[0]
            print(f"{model.name}:")
            print(f"  Prediction: {prediction}")
            print(f"  Probabilities: {probability}")

if __name__ == "__main__":
    main()