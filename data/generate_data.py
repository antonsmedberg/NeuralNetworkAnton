from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# Preprocessor Factory för att hantera olika typer av dataförberedelser
class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(data_type):
        if data_type == 'image':
            return ImagePreprocessor()
        elif data_type == 'text':
            return TextPreprocessor()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

class ImagePreprocessor:
    def preprocess(self, images):
        # Exempel på mer avancerad bildförbearbetning
        processed_images = [resize(img, (128, 128)) for img in images]  # Resize till 128x128
        processed_images = np.array(processed_images) / 255.0  # Normalisering
        return processed_images

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def preprocess(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def save_model(self, path='text_vectorizer.joblib'):
        dump(self.vectorizer, path)

    def load_model(self, path='text_vectorizer.joblib'):
        if os.path.exists(path):
            self.vectorizer = load(path)
        else:
            raise FileNotFoundError(f"{path} does not exist")


# Dynamisk val av skalare baserat på datakonfiguration
def get_scaler(scaler_type='standard'):
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

# Laddar och förbereder riktig data med möjlighet att välja preprocessor dynamiskt
def load_real_data(filepath, data_type, target_column=None, scaler_type='standard'):
    data = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    target_column = target_column or data.columns[-1]
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    preprocessor = PreprocessorFactory.get_preprocessor(data_type)
    X_preprocessed = preprocessor.preprocess(X)

    scaler = get_scaler(scaler_type)
    X_scaled = scaler.fit_transform(X_preprocessed)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Förbättrad funktion för att generera syntetisk data
def generate_synthetic_data(type='classification', n_samples=1000, n_features=20, **kwargs):
    # Här kan du utöka med ytterligare logik för att göra datagenereringen mer dynamisk
    return make_classification(n_samples=n_samples, n_features=n_features, **kwargs) if type == 'classification' else make_regression(n_samples=n_samples, n_features=n_features, **kwargs)

# Visualiserar genererad data med stöd för både klassificering och regression
def plot_generated_data(X, y, plot_type='classification'):
    plt.figure(figsize=(8, 6))
    if plot_type == 'classification':
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    elif plot_type == 'regression':
        plt.scatter(X[:, 0], y, color='r', edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2' if plot_type == 'classification' else 'Target')
    plt.title('Generated Data Visualization')
    plt.show()


if __name__ == "__main__":
    # Exempel på användning
    data_type = 'text'  # Byt till 'image' för att testa bildförprocessor
    filepath = 'path/to/your/data.csv'
    target_column = 'target'
    scaler_type = 'minmax'  # Välj 'standard' eller 'minmax'
    X_train, X_test, y_train, y_test = load_real_data(filepath, data_type, target_column, scaler_type)

    # För textförbearbetning, visa hur man sparar och laddar modellen
    if data_type == 'text':
        text_preprocessor = TextPreprocessor()
        text_preprocessor.save_model()
        text_preprocessor.load_model()

    # Demonstrera generering och visualisering av syntetisk data
    X, y = generate_synthetic_data('classification', 1000, 2, n_classes=2, random_state=42)
    plot_generated_data(X, y, 'classification')



