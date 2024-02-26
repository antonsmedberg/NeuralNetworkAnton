from models.neural_network import NeuralNetwork
from data.preprocess_data import load_data, split_data
from utils.evaluation import compute_loss  # Antag att denna funktion är definierad i evaluation.py
import numpy as np

def train_neural_network(X_train, y_train, X_val, y_val, layer_sizes, epochs=1000, learning_rate=0.001):
    """
    Tränar en neural nätverksmodell och utvärderar den på valideringsdata.

    Args:
        X_train (np.ndarray): Träningsdata egenskaper.
        y_train (np.ndarray): Träningsdata etiketter/mål.
        X_val (np.ndarray): Valideringsdata egenskaper.
        y_val (np.ndarray): Valideringsdata etiketter/mål.
        layer_sizes (list): Lista med antal neuroner i varje lager inklusive in- och utdata lager.
        epochs (int): Antal epoker för träning.
        learning_rate (float): Inlärningstakt för optimeringen.

    Returns:
        model (NeuralNetwork): Tränad modell.
        training_loss_history (list): Historik över träningsförlust per epok.
        validation_loss_history (list): Historik över valideringsförlust per epok.
    """
    model = NeuralNetwork(layer_sizes)
    model.learning_rate = learning_rate

    training_loss_history = []
    validation_loss_history = []

    for epoch in range(epochs):
        # Träning
        model.train(X_train, y_train)

        # Beräkning av förlust på träningsdata
        y_pred_train = model.predict(X_train)
        train_loss = compute_loss(y_train, y_pred_train)
        training_loss_history.append(train_loss)

        # Beräkning av förlust på valideringsdata
        y_pred_val = model.predict(X_val)
        val_loss = compute_loss(y_val, y_pred_val)
        validation_loss_history.append(val_loss)

        # Skriv ut förlustinformation varje n:te epok
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

    return model, training_loss_history, validation_loss_history

# Exempel på användning
if __name__ == "__main__":
    # Antag att load_data() laddar ditt dataset och split_data() delar upp det i tränings- och valideringsset
    X, y = load_data('path/to/your/dataset')
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)

    layer_sizes = [2, 4, 4, 1]  # Exempel på arkitektur
    model, training_loss, validation_loss = train_neural_network(X_train, y_train, X_val, y_val, layer_sizes, epochs=1000)

    # Här kan du nu spara modellen, plotta förlustkurvor etc.

