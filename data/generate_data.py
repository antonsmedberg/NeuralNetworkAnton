# generate_data.py

from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_real_data(filepath, target_column=None):
    """
    Laddar och förbereder riktig data från en angiven CSV-fil.
    Om target_column inte anges, antas den sista kolumnen vara målkolumnen.
    """
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")

    if target_column is None:
        target_column = data.columns[-1]
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dela upp data först, normalisera sedan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


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


def plot_generated_data(X, y, plot_type='classification'):
    """
    Förbättra funktionen för att stödja både klassificerings- och regressionsdata.
    """
    plt.figure(figsize=(8, 6))
    if plot_type == 'classification':
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    elif plot_type == 'regression':
        plt.scatter(X[:, 0], y, color='blue', s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2' if plot_type == 'classification' else 'Target')
    plt.title('Generated Data Visualization')
    plt.show()

def split_data(X, y, test_size=0.2, random_state=None):
    """
    Delar datan i tränings- och testset.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Användningsexempel
if __name__ == "__main__":
    X, y = generate_synthetic_data('classification', 1000, 2, 2)
    plot_generated_data(X, y, 'classification')
    # För riktig data
    X_real, y_real = load_real_data('path/to/your/data.csv')
    X_train, X_test, y_train, y_test = split_data(X_real, y_real, test_size=0.25)
