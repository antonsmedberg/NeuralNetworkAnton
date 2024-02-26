import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split

from data.generate_data import load_real_data, generate_synthetic_data, plot_generated_data
from models.neural_network import NeuralNetwork
from utils.visualization import plot_losses, plot_decision_boundary
from experiments.train_and_evaluate import train_and_evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Startar programmet...")
    try:
        # Ladda konfiguration från en JSON-fil
        config = load_config('config.json')

        # Välj datagenereringsmetod baserat på konfiguration
        if config["use_synthetic_data"]:
            logging.info("Genererar syntetisk data...")
            X, y = generate_synthetic_data('classification', n_samples=config["n_samples"], n_features=config["n_features"], n_classes=config["n_classes"], random_state=config["random_state"])
        else:
            X, y = load_real_data(config["filepath"], config.get("target_column", "target"))

        # Dela upp data i tränings- och testset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

        # Instansiera och konfigurera neurala nätverksmodellen
        model = NeuralNetwork(config["layer_sizes"], learning_rate=config["learning_rate"])

        # Träna och utvärdera modellen
        losses, accuracies = train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=config["epochs"])

        # Visualisera förlust och noggrannhet över tiden samt beslutsgränsen
        plot_losses(losses)
        if config["n_features"] == 2:
            plot_decision_boundary(model, X_test, y_test)

        logging.info("Programmet avslutat.")
    except Exception as e:
        logging.error(f"Ett fel uppstod: {e}")

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

if __name__ == '__main__':
    main()



    