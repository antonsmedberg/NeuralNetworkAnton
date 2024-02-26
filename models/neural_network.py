import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, regularization='l2', reg_lambda=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = []
        self.biases = []
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            # He-initialisering för ReLU
            if self.layer_sizes[i+1] > 1:  # Antagande: ReLU används i dolda lager
                stddev = np.sqrt(2.0 / self.layer_sizes[i])
            else:  # Xavier/Glorot-initialisering för Sigmoid
                stddev = np.sqrt(1.0 / self.layer_sizes[i])
            self.weights.append(np.random.normal(0, stddev, (self.layer_sizes[i+1], self.layer_sizes[i])))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    def relu(self, x):
        return np.maximum(0.01 * x, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def compute_loss(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2)
        if self.regularization == 'l2':
            # Lägg till L2-regulariseringsterm
            for w in self.weights:
                loss += (self.reg_lambda / 2) * np.sum(np.square(w))
        elif self.regularization == 'l1':
            # Lägg till L1-regulariseringsterm
            for w in self.weights:
                loss += self.reg_lambda * np.sum(np.abs(w))
        return loss

    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i].T) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, x, y, activations):
        deltas = [self.compute_loss(y, activations[-1]) * self.sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights))):
            delta = np.dot(deltas[-1], self.weights[i]) * self.relu_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            w_grad = np.dot(deltas[i+1].T, activations[i])
            b_grad = np.sum(deltas[i+1], axis=0, keepdims=True)
            if self.regularization == 'l2':
                w_grad += self.reg_lambda * self.weights[i]
            elif self.regularization == 'l1':
                w_grad += self.reg_lambda * np.sign(self.weights[i])
            self.weights[i] -= self.learning_rate * w_grad
            self.biases[i] -= self.learning_rate * b_grad

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)

    def predict(self, X):
        return self.forward(X)[-1]

# Exempel på användning
if __name__ == "__main__":
    nn = NeuralNetwork([2, 4, 4, 1], learning_rate=0.001, regularization='l2', reg_lambda=0.01)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    nn.train(X, y, epochs=1000)
    print(nn.predict(X))



