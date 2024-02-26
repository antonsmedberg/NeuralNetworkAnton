import matplotlib.pyplot as plt
import numpy as np

def plot_losses(losses):
    """Plottar förlustvärden över tiden."""
    plt.plot(losses)
    plt.title("Modellförlust över tiden")
    plt.xlabel("Epok")
    plt.ylabel("Förlust")
    plt.grid(True)
    plt.show()

def plot_accuracy(accuracies):
    """Plottar noggrannhet över tiden."""
    plt.plot(accuracies)
    plt.title("Modellnoggrannhet över tiden")
    plt.xlabel("Epok")
    plt.ylabel("Noggrannhet")
    plt.grid(True)
    plt.show()

def plot_decision_boundary(model, X, y):
    """Plottar beslutsgränsen för en klassificeringsmodell."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title("Beslutsgräns")
    plt.show()

def plot_feature_importances(importances, feature_names):
    """Plottar viktigheten av olika egenskaper."""
    indices = np.argsort(importances)
    plt.title('Egenskapernas viktighet')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relativ viktighet')
    plt.show()

        
