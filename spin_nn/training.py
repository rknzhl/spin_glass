import numpy as np

def train_model(model, X_train, y_train, epochs=10, learning_rate=0.01):
    """
    Обучение модели MSK.
    Args:
        model (MSKModel): Экземпляр модели.
        X_train (np.ndarray): Обучающие данные.
        y_train (np.ndarray): Метки классов.
        epochs (int): Количество эпох.
        learning_rate (float): Скорость обучения.
    Returns:
        list: Список значений функции потерь на каждой эпохе.
    """
    n_classes = len(np.unique(y_train))
    one_hot_y = np.eye(n_classes)[y_train]
    log_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in zip(X_train, one_hot_y):
            x = np.array(x, dtype=np.float32)  # Преобразуем в float32
            y = np.array(y, dtype=np.float32)

            # Прямой проход
            output = model.forward(x)

            # Потеря (кросс-энтропия)
            loss = -np.sum(y * np.log(output + 1e-8))
            epoch_loss += loss

            # Backpropagation: вычисляем градиенты
            gradients = [output - y]  # Ошибка на выходном слое
            states = [x]  # Начальное состояние

            # Прямой проход для сохранения состояний всех слоёв
            for weight in model.weights:
                local_field = np.dot(states[-1], weight)
                next_state = np.tanh(model.beta * local_field)
                states.append(next_state)

            # Распространение ошибки
            for i in range(len(model.weights) - 1, -1, -1):
                if i > 0:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - states[i]**2)
                else:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - states[0]**2)
                gradients.append(backprop_error)
            gradients.reverse()

            # Обновление весов
            for i in range(len(model.weights)):
                dw = np.outer(states[i], gradients[i+1])  # Матрица градиентов
                model.weights[i] -= learning_rate * dw

        log_losses.append(epoch_loss / len(X_train))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}")

    return log_losses