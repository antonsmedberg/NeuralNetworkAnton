from sklearn.datasets import make_classification, make_regression
import numpy as np
import matplotlib.pyplot as plt

def generate_binary_classification_data(n_samples=1000, mean_diff=[1, 1], cov=[[1, 0], [0, 1]], class_ratio=0.5):
    """
    Genererar syntetisk data för en binär klassificeringsuppgift med möjlighet att anpassa klassfördelning.

    Args:
        n_samples (int): Totalt antal prover att generera.
        mean_diff (list): Skillnad i medelvärden mellan två klasser.
        cov (list): Kovariansmatris för båda klasserna.
        class_ratio (float): Andel av den första klassen jämfört med den totala datamängden.

    Returns:
        X_mock (numpy.ndarray): Datamatris.
        y_mock (numpy.ndarray): Etikettvektor.
    """
    n_samples_class_0 = int(n_samples * class_ratio)
    n_samples_class_1 = n_samples - n_samples_class_0

    class_0 = np.random.multivariate_normal([0, 0], cov, n_samples_class_0)
    class_1 = np.random.multivariate_normal(mean_diff, cov, n_samples_class_1)

    X_mock = np.vstack((class_0, class_1))
    y_mock = np.vstack((np.zeros((n_samples_class_0, 1)), np.ones((n_samples_class_1, 1))))

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X_mock[indices], y_mock[indices]

def generate_classification_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    return make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)

def generate_regression_data(n_samples=1000, n_features=20, random_state=42):
    return make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)

def plot_generated_data(X, y):
    """
    Visualiserar genererad data.

    Args:
        X (numpy.ndarray): Datamatris.
        y (numpy.ndarray): Etikettvektor.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Generated Data Visualization')
    plt.show()

# Exempel på användning
if __name__ == "__main__":
    X_mock, y_mock = generate_binary_classification_data(n_samples=1000, mean_diff=[2, 2], class_ratio=0.6)
    plot_generated_data(X_mock, y_mock)  # Visualisera den genererade datan



