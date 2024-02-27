import numpy as np


class ActivationFunction:
    """Samling av aktiveringsfunktioner och deras derivat."""
    @staticmethod
    def relu(x, derivative=False):
        return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


class LossFunction:
    """Samling av förlustfunktioner och deras derivat."""
    @staticmethod
    def mse(y_true, y_pred, derivative=False):
        return (y_pred - y_true) / y_true.size if derivative else np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        m = y_true.shape[0]
        p = np.clip(ActivationFunction.softmax(y_pred), 1e-12, 1. - 1e-12)
        log_likelihood = -np.log(p[range(m), y_true])
        return np.sum(log_likelihood) / m

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        m = y_true.shape[0]
        grad = ActivationFunction.softmax(y_pred)
        grad[range(m), y_true] -= 1
        return grad / m


class Layer:
    """Representerar ett lager i ett neuralt nätverk."""
    def __init__(self, input_dim, output_dim, activation=None):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros((1, output_dim))
        self.activation = activation

    def activate(self, z):
        return getattr(ActivationFunction, self.activation)(z) if self.activation else z

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights.T) + self.biases
        self.output = self.activate(self.z)
        return self.output

    def backward(self, output_error, learning_rate):
        activation_derivative = getattr(ActivationFunction, f"{self.activation}_derivative", lambda x: 1)(self.z)
        activation_error = activation_derivative * output_error
        input_error = np.dot(activation_error, self.weights)
        weights_error = np.dot(activation_error.T, self.input)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(activation_error, axis=0, keepdims=True)
        return input_error


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Expected the layer to be an instance of Layer")
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred, loss='mse'):
        if not hasattr(LossFunction, loss):
            raise ValueError(f"{loss} is not a valid loss function.")
        return getattr(LossFunction, loss)(y_true, y_pred)

    def compute_loss_derivative(self, y_true, y_pred, loss='mse'):
        loss_derivative_func = f"{loss}_derivative"
        if not hasattr(LossFunction, loss_derivative_func):
            raise ValueError(f"{loss_derivative_func} is not implemented.")
        return getattr(LossFunction, loss_derivative_func)(y_true, y_pred)

    def backward(self, X, y, learning_rate, loss='mse'):
        output = self.forward(X)
        error = self.compute_loss_derivative(y, output, loss)
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, X, y, epochs, learning_rate, loss='mse', batch_size=None):
        if batch_size is None or batch_size >= len(X):
            self._train_full_batch(X, y, epochs, learning_rate, loss)
        else:
            self._train_mini_batches(X, y, epochs, learning_rate, loss, batch_size)

    def _train_full_batch(self, X, y, epochs, learning_rate, loss):
        for epoch in range(epochs):
            self.backward(X, y, learning_rate, loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.compute_loss(y, self.forward(X), loss)}")

    def _train_mini_batches(self, X, y, epochs, learning_rate, loss, batch_size):
        for epoch in range(epochs):
            for start_idx in range(0, len(X), batch_size):
                end_idx = start_idx + batch_size
                batch_X, batch_y = X[start_idx:end_idx], y[start_idx:end_idx]
                self.backward(batch_X, batch_y, learning_rate, loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.compute_loss(y, self.forward(X), loss)}")

    def predict(self, X):
        return self.forward(X)



# Exempel på användning med kommentarer för att guida användaren.
if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 4, 'relu'))
    nn.add_layer(Layer(4, 4, 'relu'))
    nn.add_layer(Layer(4, 1, 'sigmoid'))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn.train(X, y, epochs=2000, learning_rate=0.1, loss='mse')
    print("Predictions:", nn.predict(X))






