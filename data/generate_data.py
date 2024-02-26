from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_real_data(filepath, target_column=None):
    """
    Laddar och förbereder riktig data från en angiven fil.
    Stöder CSV och Excel-filer. Normaliserar funktionerna och returnerar delade tränings- och testdata.

    Args:
        filepath (str): Sökväg till filen som innehåller data.
        target_column (str, optional): Namnet på målkolumnen. Använder sista kolumnen som default.

    Returns:
        tuple: Fyra numpy arrays: X_train_scaled, X_test_scaled, y_train, y_test
    """
    # Läs data beroende på filformat
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    else:
        raise ValueError("File format not supported. Only .csv and .xlsx are accepted.")

    # Använd sista kolumnen som målkolumn om ingen ges
    if target_column is None:
        target_column = data.columns[-1]
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Dela upp data och normalisera
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

def generate_synthetic_data(type='classification', n_samples=1000, n_features=20, **kwargs):
    """
    Genererar och returnerar syntetisk data baserat på specifierad typ.

    Args:
        type (str): 'classification' eller 'regression' för datatyp.
        n_samples (int): Antal sampel att generera.
        n_features (int): Antal funktioner för varje sampel.
        **kwargs: Extra argument som skickas till genereringsfunktionen.

    Returns:
        tuple: Två numpy arrays: X (funktionerna), y (målet).
    """
    if type == 'classification':
        return make_classification(n_samples=n_samples, n_features=n_features, **kwargs)
    elif type == 'regression':
        return make_regression(n_samples=n_samples, n_features=n_features, **kwargs)
    else:
        raise ValueError("Invalid type specified. Use 'classification' or 'regression'.")

def plot_generated_data(X, y, plot_type='classification'):
    """
    Visualiserar genererad data. Stöder både klassificerings- och regressionsdata.

    Args:
        X (numpy.ndarray): Datamatris.
        y (numpy.ndarray): Etikettvektor.
        plot_type (str): 'classification' eller 'regression' för att bestämma plot-typen.
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

if __name__ == "__main__":
    # Demonstrerar generering och visualisering av syntetisk data
    X, y = generate_synthetic_data('classification', 1000, 2, n_classes=2, random_state=42)
    plot_generated_data(X, y, 'classification')

