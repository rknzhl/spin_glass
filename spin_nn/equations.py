import numpy as np
from spin_nn.model import MSKModel
from concurrent.futures import ThreadPoolExecutor

def l2_hinge_loss(predictions, targets):
    """
    L2-многоклассовый hinge loss: 
    loss = sum_{j != y_true}(max(0, 1 - pred[y_true] + pred[j]))^2

    predictions: shape (batch, num_classes)
    targets: one-hot shape (batch, num_classes)
    """
    # Индексы правильных классов
    true_indices = np.argmax(targets, axis=1)
    batch_size = predictions.shape[0]
    loss = 0.0
    for i in range(batch_size):
        true_class = true_indices[i]
        margin_losses = []
        for j in range(predictions.shape[1]):
            if j != true_class:
                margin = max(0, 1 - predictions[i, true_class] + predictions[i, j])
                margin_losses.append(margin**2)
        loss += sum(margin_losses)
    return loss / batch_size


def calculate_epoch_energy(model, X_train, beta=1.0):
    total_energy = 0.0
    for x in X_train:
        energy = model.calculate_hamiltonian(x)
        total_energy += energy

    return total_energy / len(X_train)


def compute_critical_temperature(weights):
    """
    Вычисляет критическую температуру (температуру Кюри) для данной модели.
    """
    # Определяем размер полной матрицы взаимодействий
    total_size = sum(w.shape[0] for w in weights) + weights[-1].shape[1]
    interaction_matrix = np.zeros((total_size, total_size))

    # Заполняем блоки матрицы взаимодействий
    offset = 0
    for i, weight in enumerate(weights):
        rows, cols = weight.shape
        interaction_matrix[offset:offset+rows, offset:offset+cols] = weight
        interaction_matrix[offset:offset+cols, offset:offset+rows] = weight.T
        offset += rows

    # Вычисляем собственные значения
    eigenvalues = np.linalg.eigvals(interaction_matrix)

    # Находим максимальное собственное значение
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Если максимальное собственное значение равно 0, система неустойчива
    if max_eigenvalue == 0:
        raise ValueError("Invalid interaction matrix: maximum eigenvalue is zero.")

    # Температура Кюри: обратная пропорция к максимальному собственному значению
    Tc = 1.0 / max_eigenvalue

    return Tc