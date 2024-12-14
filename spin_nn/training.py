import numpy as np



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
        x = np.array(x, dtype=np.float32)  # Убедимся, что x имеет правильный формат
        output = model.forward(x)  # Прямой проход через модель
        
        # Предсказание: выбираем класс с максимальной вероятностью
        predicted_class = np.argmax(output)
        true_class = y
        
        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy



def train_model(model, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.001):
    """
    Обучение модели MSK.
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

            # Проверка выходных значений
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                print(f"Invalid output detected: {output}")
                return log_losses

            # Потеря (кросс-энтропия)
            output = np.clip(output, 1e-8, 1 - 1e-8)  # Предотвращаем логарифм нуля
            loss = -np.sum(y * np.log(output))
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
                dw = np.outer(states[i], gradients[i + 1])  # Вычисляем матрицу градиентов
                model.weights[i] -= learning_rate * dw

        log_losses.append(epoch_loss / len(X_train))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train):.4f}")

        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy * 100:.2f}%")

    return log_losses