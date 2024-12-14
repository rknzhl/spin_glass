import numpy as np

class MSKModel:
    def __init__(self, n_layers, layer_sizes, beta=1.0):
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = [
            np.random.normal(0, 1 / np.sqrt(self.layer_sizes[i]), 
                             size=(self.layer_sizes[i], self.layer_sizes[i+1]))
            for i in range(self.n_layers - 1)
        ]
        return weights
    

    def forward(self, spins):
        """
        Прямой проход через слои с учётом TAP-поправок.
        Args:
            spins (np.ndarray): Входные данные (стартовые состояния спинов).
        Returns:
            np.ndarray: Состояния выходного слоя.
        """
        # Убедимся, что входные данные числовые
        spins = np.array(spins, dtype=np.float32)
        states = [spins]
        
        for i, weight in enumerate(self.weights):
            # Локальное магнитное поле
            local_field = np.dot(states[-1], weight)
            
            # Онсагеровская поправка (TAP)
            onsager_correction = self.beta**2 * np.sum(weight**2, axis=0) * (1 - np.mean(states[-1]**2))
            
            # Применяем tanh с учётом онсагеровской поправки
            next_state = np.tanh(self.beta * local_field - onsager_correction)
            states.append(next_state)
        
        return states[-1]
    
