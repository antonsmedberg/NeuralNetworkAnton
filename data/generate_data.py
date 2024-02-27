import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import os
from PIL import Image

# Baspreprocessor
class BasePreprocessor:
    def preprocess(self, data):
        raise NotImplementedError("Subclass must implement abstract method")

# Bildförprocessor
class ImagePreprocessor(BasePreprocessor):
    def preprocess(self, images):
        processed_images = [self.resize_image(img, (128, 128)) for img in images]
        processed_images = np.array(processed_images) / 255.0
        return processed_images

    @staticmethod
    def resize_image(image, size):
        return Image.open(image).resize(size, Image.ANTIALIAS)

# Textförprocessor
class TextPreprocessor(BasePreprocessor):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def preprocess(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def save_model(self, path='text_vectorizer.joblib'):
        dump(self.vectorizer, path)

    def load_model(self, path='text_vectorizer.joblib'):
        self.vectorizer = load(path) if os.path.exists(path) else None

# Välj preprocessor
def get_preprocessor(data_type):
    preprocessors = {
        'image': ImagePreprocessor(),
        'text': TextPreprocessor(),
    }
    return preprocessors.get(data_type, BasePreprocessor())

# Välj skalare
def get_scaler(scaler_type):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
    }
    return scalers.get(scaler_type, StandardScaler())

# Ladda och förbered riktig data
def load_real_data(filepath, data_type, target_column=None, scaler_type='standard'):
    data = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    X = data.drop(columns=target_column) if target_column else data.iloc[:, :-1]
    y = data[target_column] if target_column else data.iloc[:, -1]

    preprocessor = get_preprocessor(data_type)
    X_preprocessed = preprocessor.preprocess(X)

    scaler = get_scaler(scaler_type)
    X_scaled = scaler.fit_transform(X_preprocessed)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Generera syntetisk data
def generate_synthetic_data(type='classification', n_samples=1000, n_features=20, **kwargs):
    return make_classification(n_samples=n_samples, n_features=n_features, **kwargs) if type == 'classification' else make_regression(n_samples=n_samples, n_features=n_features, **kwargs)

# Visualisera genererad data
def plot_generated_data(X, y, plot_type='classification'):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis' if plot_type == 'classification' else 'coolwarm', edgecolors='k', s=20)
    plt.colorbar(scatter)
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



