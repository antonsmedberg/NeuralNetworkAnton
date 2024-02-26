from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_real_data(filepath, target_column='target'):
    """
    Laddar och förbereder riktig data från en angiven CSV-fil.

    Args:
        filepath (str): Sökvägen till CSV-filen som innehåller datan.
        target_column (str): Namnet på kolumnen som innehåller målvärdet.

    Returns:
        X (numpy.ndarray): Förklarande variabler (funktioner) i datan.
        y (numpy.ndarray): Målvärdet (etiketter) i datan.
    """
    data = pd.read_csv(filepath)
    X = data.drop(target_column, axis=1).values
    y = data[target_column].values

    # Normalisera datan
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def generate_synthetic_data(type='classification', n_samples=1000, n_features=20, **kwargs):
    """
    Genererar syntetisk data för antingen klassificering eller regression.

    Args:
        type (str): Typ av data att generera ('classification' eller 'regression').
        n_samples (int): Antalet prover att generera.
        n_features (int): Antalet funktioner för varje prov.
        **kwargs: Ytterligare argument beroende på datatyp.

    Returns:
        X (numpy.ndarray): Förklarande variabler (funktioner) i den genererade datan.
        y (numpy.ndarray): Målvärdet (etiketter) i den genererade datan.
    """
    if type == 'classification':
        return make_classification(n_samples=n_samples, n_features=n_features, **kwargs)
    elif type == 'regression':
        return make_regression(n_samples=n_samples, n_features=n_features, **kwargs)
    else:
        raise ValueError("Ogiltig typ specifierad. Använd 'classification' eller 'regression'.")

def plot_generated_data(X, y):
    """
    Visualiserar genererad data.

    Args:
        X (numpy.ndarray): Datamatris.
        y (numpy.ndarray): Etikettvektor.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Generated Data Visualization')
    plt.show()

# Exempel på användning
if __name__ == "__main__":
    # Exempel på hur man genererar och visualiserar syntetisk klassificeringsdata
    X_class, y_class = generate_synthetic_data(type='classification', n_samples=1000, n_features=2, n_classes=2, random_state=42)
    plot_generated_data(X_class, y_class)

    # Exempel på hur man laddar och förbereder riktig data
    # X_real, y_real = load_real_data('path/to/your/data.csv', target_column='your_target_column')
