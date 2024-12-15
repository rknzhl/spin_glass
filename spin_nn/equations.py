import numpy as np
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

