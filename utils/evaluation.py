import numpy as np

def compute_loss(y_true, y_pred):
    """Beräknar genomsnittlig kvadratisk förlust."""
    return np.mean((y_true - y_pred) ** 2)

def compute_accuracy(y_true, y_pred):
    """Beräknar noggrannheten för klassificeringsuppgifter."""
    predictions = np.round(y_pred)  # Anta binär klassificering
    correct = (predictions == y_true)
    accuracy = np.mean(correct)
    return accuracy

# Du kan lägga till fler utvärderingsfunktioner här, t.ex. för precision och recall
