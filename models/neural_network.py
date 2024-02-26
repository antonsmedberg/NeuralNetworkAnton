import numpy as np

class NeuralNetwork:
    """
    En enkel neural nätverksklass för binär klassificering eller regression.
    """
    def __init__(self, layer_sizes, learning_rate=0.001, regularization='l2', reg_lambda=0.01):
        """
        Initierar nätverket med givna parametrar.

        Args:
            layer_sizes (list): Lista med antalet noder i varje lager, inklusive in- och utdatan.
            learning_rate (float): Inlärningshastigheten.
            regularization (str): Typ av regularisering ('l2' eller 'l1').
            reg_lambda (float): Regulariseringsparameter.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.init_weights()

    def init_weights(self):
        """
        Initialiserar vikterna och biaserna med He eller Xavier/Glorot-initialisering
        baserat på aktiveringsfunktionen.
        """
        for i in range(len(self.layer_sizes) - 1):
            if self.layer_sizes[i+1] > 1:  # Antagande: ReLU används
                stddev = np.sqrt(2.0 / self.layer_sizes[i])
            else:  # Antagande: Sigmoid används för det sista lagret
                stddev = np.sqrt(1.0 / self.layer_sizes[i])
            self.weights.append(np.random.normal(0, stddev, (self.layer_sizes[i+1], self.layer_sizes[i])))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    # Aktiveringsfunktioner och deras derivat
    def relu(self, x):
        return np.maximum(0.01 * x, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def compute_loss(self, y_true, y_pred):
        """
        Beräknar förlusten med vald regularisering.
        """
        loss = np.mean((y_true - y_pred) ** 2)
        # Regulariseringstermer
        if self.regularization == 'l2':
            for w in self.weights:
                loss += (self.reg_lambda / 2) * np.sum(np.square(w))
        elif self.regularization == 'l1':
            for w in self.weights:
                loss += self.reg_lambda * np.sum(np.abs(w))
        return loss

    def forward(self, x):
        """
        Utför ett framåtpass genom nätverket.
        """
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i]
            a = self.relu(z) if i < len(self.weights) - 1 else self.sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, x, y, activations):
        """
        Utför bakåtpass genom nätverket och uppdaterar vikterna och biaserna.
        """
        deltas = [self.compute_loss(y, activations[-1]) * self.sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights))):
            delta = np.dot(deltas[-1], self.weights[i]) * self.relu_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()

        # Uppdaterar vikter och biaser
        for i in range(len(self.weights)):
            w_grad = np.dot(deltas[i+1].T, activations[i])
            b_grad = np.sum(deltas[i+1], axis=0, keepdims=True)
            # Tillämpar regularisering om så krävs
            if self.regularization == 'l2':
                w_grad += self.reg_lambda * self.weights[i]
            elif self.regularization == 'l1':
                w_grad += self.reg_lambda * np.sign(self.weights[i])
            self.weights[i] -= self.learning_rate * w_grad
            self.biases[i] -= self.learning_rate * b_grad

    def train(self, X, y, epochs=10000, batch_size=32):
        """
        Tränar nätverket med angivet antal epoker och batch-storlek.
        """
        for epoch in range(epochs):
            for X_batch, y_batch in self.generate_batches(X, y, batch_size):
                activations = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations)

    def generate_batches(self, X, y, batch_size):
        """
        Generator för att skapa batcher av data.
        """
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size], y[i:i + batch_size])

    def predict(self, X):
        """
        Gör förutsägelser med nätverket för givet X.
        """
        return self.forward(X)[-1]

# Exempel på användning
if __name__ == "__main__":
    nn = NeuralNetwork([2, 4, 4, 1], learning_rate=0.001, regularization='l2', reg_lambda=0.01)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    nn.train(X, y, epochs=1000)
    print(nn.predict(X))




