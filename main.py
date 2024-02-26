import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.generate_data import generate_binary_classification_data
from models.neural_network import NeuralNetwork
from utils.visualization import plot_losses, plot_decision_boundary
from experiments.train_and_evaluate import train_and_evaluate

# Konfigurera loggning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_real_data(filepath):
    """
    Laddar och förbereder riktig data från en angiven CSV-fil.
    """
    try:
        logging.info("Laddar riktig data från {}".format(filepath))
        data = pd.read_csv(filepath)
        X = data.drop('target_column', axis=1)  # Byt ut 'target_column' mot den faktiska målkolumnen.
        y = data['target_column']

        # Normalisera datan
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y.values

    except FileNotFoundError:
        logging.error(f"Fil {filepath} hittades inte.")
        exit()
    except Exception as e:
        logging.error(f"Ett fel uppstod vid laddning av data: {e}")
        exit()



def main():
    logging.info("Startar programmet...")

    # Ladda konfiguration från en JSON-fil
    config = load_config('config.json')

    # Välj datagenereringsmetod baserat på konfiguration
    if config["use_synthetic_data"]:
        logging.info("Genererar syntetisk data...")
        X, y = generate_binary_classification_data(
            n_samples=config["n_samples"], 
            n_features=config["n_features"], 
            n_classes=config["n_classes"], 
            random_state=config["random_state"])
    else:
        X, y = load_real_data(config["filepath"])
    
    # Dela upp data i tränings- och testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

    # Instansiera och konfigurera neurala nätverksmodellen
    model = NeuralNetwork(config["layer_sizes"], learning_rate=config["learning_rate"])

    # Träna och utvärdera modellen
    losses, accuracies = train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=config["epochs"])

    # Visualisera förlust och noggrannhet över tiden
    plot_losses(losses)
    # Observera: Antag att plot_accuracy funktionen finns. Ersätt eller lägg till relevant kod
    # plot_accuracy(accuracies)

    # Plotta beslutsgränsen om datan är 2D
    if config["n_features"] == 2:
        plot_decision_boundary(model, X_test, y_test)

    logging.info("Programmet avslutat.")

def load_config(config_path):
    """
    Laddar konfigurationsinställningar från en JSON-fil.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

if __name__ == '__main__':
    main()


    