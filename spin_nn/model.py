import numpy as np
import json
class MSKModel:
    def __init__(self, layer_sizes, lr=0.01, momentum=0.9):
        """
        Модель с бинаризованными скрытыми слоями.
        layer_sizes: список размеров слоёв, например [784, 512, 512, 10]
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = [
            np.random.normal(0, 1 / np.sqrt(self.layer_sizes[i]),
                             size=(self.layer_sizes[i], self.layer_sizes[i+1]))
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.velocity = [np.zeros_like(w) for w in self.weights]

    @staticmethod
    def step_function(x):
        """Прямая активация: binarization step."""
        return np.where(x >= 0, 1, -1)

    @staticmethod
    def h_tanh_grad(x):
        """
        Производная для обратного прохода (STE).
        h_tanh(x) определена кусочно: -1 если x<-1, x если -1<x<1, 1 если x>1
        Производная этой функции равна 1 в диапазоне (-1,1) и 0 вне его.
        """
        #if (np.abs(x)<1):
        #    return x;
        #elif (x < -1):
        #    return -1;
        #else:
        #    return 1;
        return np.where(np.abs(x) < 1, 1.0, 0.0)

    def forward(self, x):
        """
        Прямой проход.
        Возвращает (output, activations, pre_activations),
        где activations[i] - активации слоя i,
              pre_activations[i] - доактивационные значения слоя i.
        """
        activations = [x]    # входной слой без изменений
        pre_activations = [] # доактивации скрытых слоёв

        # Проход по скрытым слоям (бинаризация)
        for i in range(len(self.weights)-1):
            z = np.dot(activations[-1], self.weights[i]) # доактивация
            pre_activations.append(z)
            a = self.step_function(z)  # бинаризация
            activations.append(a)

        # Последний слой - линейный выход
        z = np.dot(activations[-1], self.weights[-1]) # доактивация выходного слоя (линейная)
        pre_activations.append(z)
        output = self.step_function(z) ## ДЕЛАЕМ БИНАРИЗАЦИЮ
        activations.append(z)

        return activations[-1], activations, pre_activations

    def backward(self, output, y_true, activations, pre_activations):
        """
        Обратное распространение.
        output: выход сети (последний слой)
        y_true: целевые значения one-hot
        activations: список активаций
        pre_activations: список доактиваций

        Возвращает список градиентов для всех весов.
        """

        # Градиенты по весам
        gradients = [np.zeros_like(w) for w in self.weights]

        # Ошибка на выходном слое
        delta = (output - y_true)  # (num_classes,)

        # Градиент для последнего слоя
        # activations[-2] - это активации последнего скрытого слоя
        gradients[-1] = np.outer(activations[-2], delta)

        # Обратный проход по скрытым слоям
        # pre_activations[-1] соответствует выходному слою, последний скрытый слой - pre_activations[-2]
        # Нам нужно пройти по слоям с конца к началу, пропуская выходной, для него уже сделан grad
        for i in range(len(self.weights) - 2, -1, -1):
            # delta для текущего слоя
            # Применяем веса следующего слоя
            delta = np.dot(delta, self.weights[i+1].T)

            # Если это не последний слой (который линейный), то применяем h_tanh_grad
            # Последний слой в pre_activations - это выходной (линейный),
            # поэтому для всех кроме последнего слоя:
            if i < len(self.weights)-1: 
                # Применяем STE производную
                delta = delta * self.h_tanh_grad(pre_activations[i])

            # Градиент для текущего слоя
            gradients[i] = np.outer(activations[i], delta)

        return gradients

    def update_weights(self, avg_gradients):
        for i in range(len(self.weights)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * avg_gradients[i]
            self.weights[i] += self.velocity[i]

    def calculate_hamiltonian(self, spins):
        energy = 0
        current_spins = spins
        for weight in self.weights:
            # Локальное поле для текущего слоя
            local_field = np.dot(current_spins, weight)  # Размерность соответствует следующему слою

            # Обновление энергии
            energy -= 0.5 * np.sum(local_field * current_spins[:local_field.shape[0]])

            # Обновление состояний спинов
            current_spins = self.step_function(local_field)  # Бинаризация
            
        return energy
    
    def save_weights(self, filepath):
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights]
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        print(f"Weights and structure saved to {filepath}")

    @classmethod
    def load_weights(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        layer_sizes = data["layer_sizes"]
        model = cls(layer_sizes=layer_sizes)

        # Восстанавливаем веса
        model.weights = [np.array(w) for w in data["weights"]]
        print(f"Weights and structure loaded from {filepath}")

        return model