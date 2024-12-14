import numpy as np

def train_model(model, X_train, y_train, epochs=10, learning_rate=0.01):
    """
    Обучение модели MSK с TAP-поправками.
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
            x = np.array(x, dtype=np.float32)  # Убедимся, что x — float32
            y = np.array(y, dtype=np.float32)

            # Forward: получаем состояния выходного слоя
            output = model.forward(x)

            # Потеря (кросс-энтропия)
            loss = -np.sum(y * np.log(output + 1e-8))
            epoch_loss += loss

            # Backpropagation: вычисляем градиенты
            gradients = [output - y]  # Градиент на выходном слое
            for i in range(len(model.weights) - 1, -1, -1):  # Проходим слои в обратном порядке
                if i > 0:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - model.weights[i - 1]**2)
                else:
                    backprop_error = gradients[-1].dot(model.weights[i].T) * (1 - x**2)
                gradients.append(backprop_error)
            gradients.reverse()

            # Обновляем веса
            for i in range(len(model.weights)):
                dw = np.outer(x if i == 0 else model.weights[i-1], gradients[i])
                model.weights[i] -= learning_rate * dw

        log_losses.append(epoch_loss / len(X_train))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}")

    return log_losses