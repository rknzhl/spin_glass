import numpy as np

def tap_free_energy(weights, spins, beta):
    """
    Вычисление TAP-свободной энергии для многослойной сети.
    
    Args:
        weights (list of np.ndarray): Список матриц весов между слоями.
        spins (np.ndarray): Состояния спинов начального слоя.
        beta (float): Обратная температура (1 / T).
    
    Returns:
        float: TAP свободная энергия.
    """
    H = 0
    current_spins = spins

    for weight in weights:
        # Локальное взаимодействие: вычисляем локальное поле для текущего слоя
        local_field = np.dot(current_spins, weight)  # Размер (нейроны текущего слоя,)
        
        # Энергия вклада текущего слоя
        H -= 0.5 * np.sum(local_field**2)
        
        # Обновляем состояния спинов для следующего слоя
        current_spins = np.tanh(beta * local_field)

    # TAP-поправка (учёт флуктуаций)
    onsager_term = -0.5 * beta * np.sum(1 - spins**2)
    
    return H + onsager_term


def evaluate_model(model, X_test, y_test):
    """
    Оценка точности модели на тестовой выборке.
    
    Args:
        model (MSKModel): Обученная модель.
        X_test (np.ndarray): Тестовые данные.
        y_test (np.ndarray): Метки классов для тестовых данных.
    
    Returns:
        float: Точность модели (accuracy).
    """
    correct_predictions = 0
    total_predictions = len(X_test)
    
    for x, y in zip(X_test, y_test):
        x = np.array(x, dtype=np.float32)
        output = model.forward(x)
        predicted_class = np.argmax(output)
        true_class = y
        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def train_model(model, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.001):
    """
    Обучение модели MSK с функцией потерь на основе TAP-свободной энергии.
    
    Args:
        model (MSKModel): Обучаемая модель.
        X_train (np.ndarray): Обучающие данные.
        y_train (np.ndarray): Метки классов для обучения.
        X_test (np.ndarray): Тестовые данные.
        y_test (np.ndarray): Метки классов для теста.
        epochs (int): Количество эпох обучения.
        learning_rate (float): Скорость обучения.
    
    Returns:
        list: Список значений функции потерь по эпохам.
    """
    n_classes = len(np.unique(y_train))
    one_hot_y = np.eye(n_classes)[y_train]
    log_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in zip(X_train, one_hot_y):
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # Прямой проход
            output = model.forward(x)

            # TAP-свободная энергия
            tap_energy = tap_free_energy(model.weights, x, model.beta)
            
            # Кросс-энтропийная потеря
            output = np.clip(output, 1e-8, 1 - 1e-8)
            cross_entropy_loss = -np.sum(y * np.log(output))
            
            # Общая потеря = TAP + Cross-Entropy
            loss = tap_energy + cross_entropy_loss
            epoch_loss += loss

            # Backpropagation
            gradients = [output - y]
            states = [x]
            for weight in model.weights:
                local_field = np.dot(states[-1], weight)
                next_state = np.tanh(model.beta * local_field)
                states.append(next_state)

            for i in range(len(model.weights) - 1, -1, -1):
                if i > 0:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - states[i]**2)
                else:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - states[0]**2)
                gradients.append(backprop_error)
            gradients.reverse()

            for i in range(len(model.weights)):
                dw = np.outer(states[i], gradients[i + 1])
                model.weights[i] -= learning_rate * dw

        log_losses.append(epoch_loss / len(X_train))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}")

        # Оценка точности на тестовой выборке
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy * 100:.2f}%")

    return log_losses