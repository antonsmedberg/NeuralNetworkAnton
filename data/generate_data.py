from sklearn.datasets import make_classification, make_regression
import numpy as np

def generate_synthetic_classification(n_samples=1000, n_features=20, n_classes=2, noise=0.05):
    # Generera syntetisk klassificeringsdata
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_clusters_per_class=1, flip_y=noise, random_state=42)
    return X, y

def generate_synthetic_regression(n_samples=1000, n_features=20, noise=1.0):
    # Generera syntetisk regressionsdata
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

# Du kan lägga till fler funktioner för att generera andra typer av data eller med olika parametrar
