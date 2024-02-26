import numpy as np
from data.generate_data import generate_binary_classification_data
from models.neural_network import NeuralNetwork
from utils.visualization import plot_losses, plot_decision_boundary
from experiments.train_and_evaluate import train_and_evaluate
from sklearn.model_selection import train_test_split


def load_real_data(filepath):
    """
    Laddar riktig data från en CSV-fil.

    Args:
        filepath (str): Sökväg till CSV-filen med riktig data.

    Returns:
        numpy.ndarray: Egenskaper (X).
        numpy.ndarray: Mål/etiketter (y).
    """
    import pandas as pd
    data = pd.read_csv(filepath)
    X = data.drop('target_column', axis=1).values  # Byt 'target_column' mot namnet på din målkolumn
    y = data['target_column'].values
    return X, y


def main():
    # Konfiguration
    config = {
        "use_synthetic_data": True,  # Byt till False när du vill använda riktig data
        "filepath": "path/to/your/real/data.csv",  # Ange sökvägen till din riktiga data
        
        "layer_sizes": [2, 4, 4, 1],
        "epochs": 1000,
        "learning_rate": 0.001,
        "test_size": 0.2,
        "n_samples": 1000,  # Antal prover för syntetisk data
        "n_features": 2,  # Antal egenskaper för att möjliggöra beslutsgränsvisualisering
        "n_classes": 2,  # För binär klassificering
        "random_state": 42
        # Lägg till fler konfigurationsparametrar vid behov
    }

    # Välj datagenereringsmetod baserat på konfiguration
    if config["use_synthetic_data"]:
        X, y = generate_binary_classification_data(
            n_samples=config["n_samples"], 
            n_features=config["n_features"], 
            n_classes=config["n_classes"], 
            random_state=config["random_state"])
    else:
        X, y = load_real_data(config["filepath"])
    
    # Dela upp data i tränings- och testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
    

    # Generera syntetisk data
    X, y = generate_binary_classification_data(n_samples=config["n_samples"], n_features=config["n_features"], n_classes=config["n_classes"], random_state=config["random_state"])
    model = NeuralNetwork(config["layer_sizes"], learning_rate=config["learning_rate"])

    # Träna och utvärdera modellen
    losses, accuracies = train_and_evaluate(model, X, y, epochs=config["epochs"], test_size=config["test_size"])

    # Visualisera förlust och noggrannhet över tiden
    plot_losses(losses)
    plot_accuracy(accuracies)  # Antag att denna funktion finns i visualization.py
    plot_decision_boundary(model, X_test, y_test)  # Notera: Funktionen kan behöva anpassas för din modells predict-metod

    # Plotta beslutsgränsen om datan är 2D
    if config["n_features"] == 2:
        plot_decision_boundary(model, X_test, y_test)

if __name__ == '__main__':
    main()


    