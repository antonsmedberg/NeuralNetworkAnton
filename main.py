import numpy as np
import pandas as pd
import json
import logging
from data.generate_data import load_real_data, generate_synthetic_data, plot_generated_data
from models.neural_network import NeuralNetwork
from utils.visualization import plot_losses, plot_decision_boundary
from experiments.train_and_evaluate import train_and_evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_config(config):
    required_keys = ["use_synthetic_data", "n_samples", "n_features", "n_classes", "random_state", "layer_sizes", "learning_rate", "epochs", "test_size"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(config[key], (int, float)) or (isinstance(config[key], int) and config[key] < 0):
            raise ValueError(f"Invalid value for {key}: {config[key]}")
        if not isinstance(config["use_synthetic_data"], bool):
            raise ValueError("use_synthetic_data must be a boolean")
        

def main():
    logging.info("Startar programmet...")
    try:
        config = load_config('config.json')
        validate_config(config)

        preprocessor = PreprocessorFactory.get_preprocessor(config["data_type"])
        handler = OutputHandlerFactory.get_handler(config["task_type"])

        if config["use_synthetic_data"]:
            logging.info("Genererar syntetisk data...")
            raw_data = generate_synthetic_data(...)
            X, y = preprocessor.preprocess(raw_data)
        else:
            raw_data = load_data(config["filepath"])
            X, y = preprocessor.preprocess(raw_data)

        model = NeuralNetwork(config["layer_sizes"], learning_rate=config["learning_rate"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
        losses, accuracies = train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=config["epochs"])

        predictions = model.predict(X_test)
        final_output = handler.handle_output(predictions)

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




    