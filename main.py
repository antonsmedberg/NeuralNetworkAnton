import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from data.generate_data import load_real_data, generate_synthetic_data
from models.neural_network import NeuralNetwork
from utils.visualization import plot_losses, plot_decision_boundary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigurationValidator:
    @staticmethod
    def validate_config(config):
        required_keys = ["use_synthetic_data", "n_samples", "n_features", "n_classes", "random_state", "layer_sizes", "learning_rate", "epochs", "test_size"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            if not isinstance(config[key], (int, float)) or (isinstance(config[key], int) and config[key] < 0):
                raise ValueError(f"Invalid value for {key}: {config[key]}")
            if not isinstance(config["use_synthetic_data"], bool):
                raise ValueError("use_synthetic_data must be a boolean")

def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    ConfigurationValidator.validate_config(config)
    return config

def main():
    logging.info("Startar programmet...")
    try:
        config = load_config()

        if config["use_synthetic_data"]:
            logging.info("Genererar syntetisk data...")
            X, y = generate_synthetic_data(n_samples=config["n_samples"], n_features=config["n_features"], n_classes=config["n_classes"], random_state=config["random_state"])
        else:
            X, y = load_real_data(config["filepath"], target_column=config.get("target_column"))

        model = NeuralNetwork(config["layer_sizes"], learning_rate=config["learning_rate"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
        model.train(X_train, y_train, epochs=config["epochs"])

        plot_losses(model.losses)
        if config["n_features"] == 2:
            plot_decision_boundary(model, X_test, y_test)

        logging.info("Programmet avslutat.")
    except Exception as e:
        logging.error(f"Ett fel uppstod: {e}")

if __name__ == '__main__':
    main()




def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

if __name__ == '__main__':
    main()




    