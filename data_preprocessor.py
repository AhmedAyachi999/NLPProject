import pandas as pd


class DataPreprocessor:

    def __init__(self, file_path):
        # inside __init__
        self.text_columns = [
            "AVO-Kurztext",
            "AVO-Langtext",
            "VD-Kurztext",
            "Beschreibung",
            "Kurzcode"
        ]
        self.file_path = file_path
        self.X = None
        self.y = None


    def load_and_preprocess(self):
        df = pd.read_csv(self.file_path, sep=',')
        # Nur benötigte Spalten
        df = df[self.text_columns + ["label"]]
        for col in self.text_columns:
            df[col] = df[col].fillna("")
        df['all_text'] = df[self.text_columns].agg(' '.join, axis=1)
        self.X = df['all_text']
        self.y = df['label'].apply(lambda x: 1 if x != 0 else 0)
        return self.X, self.y
