import numpy as np

class MSKModel:
    def __init__(self, layer_sizes, beta=1.0):
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.n_layers = len(layer_sizes)
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = [
            np.random.normal(0, 1 / np.sqrt(self.layer_sizes[i]),
                             size=(self.layer_sizes[i], self.layer_sizes[i+1]))
            for i in range(self.n_layers - 1)
        ]
        return weights

    def smoothed_step(self, x):
        """
        Сглаженная бинаризация (аналог активации).
        """
        return np.tanh(self.beta * x)

    def forward(self, spins):
        """
        Прямой проход через сеть.
        """
        current_spins = spins
        for weight in self.weights[:-1]:
            local_field = np.dot(current_spins, weight)
            current_spins = self.smoothed_step(local_field)

        final_weight = self.weights[-1]
        local_field = np.dot(current_spins, final_weight)
        output = np.exp(local_field - np.max(local_field))
        return output / np.sum(output)