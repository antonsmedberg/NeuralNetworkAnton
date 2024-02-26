
from models.neural_network import NeuralNetwork
from utils.evaluation import compute_accuracy, compute_loss
import numpy as np

def train_and_evaluate(model, X_train, y_train, X_val=None, y_val=None, epochs=1000):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):
        # Träna modellen på träningsdatan
        model.train(X_train, y_train)

        # Beräkna träningsförlust
        predictions_train = model.predict(X_train)
        train_loss = compute_loss(y_train, predictions_train)
        train_losses.append(train_loss)

        # Validera modellen om valideringsdata finns
        if X_val is not None and y_val is not None:
            predictions_val = model.predict(X_val)
            val_loss = compute_loss(y_val, predictions_val)
            val_losses.append(val_loss)

            # Beräkna noggrannhet för valideringssetet
            accuracy = compute_accuracy(y_val, predictions_val)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {accuracy}")
        else:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

    return train_losses, val_losses, accuracies
