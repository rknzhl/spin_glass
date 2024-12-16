import numpy as np
from tqdm import tqdm
from spin_nn.equations import l2_hinge_loss, calculate_epoch_energy
from concurrent.futures import ThreadPoolExecutor

def evaluate(model, X_test, y_test):
    correct = 0
    for x, y in zip(X_test, y_test):
        output, _, _ = model.forward(x)
        if np.argmax(output) == np.argmax(y):
            correct += 1
    return 100.0 * correct / len(X_test)

def train_on_batch(model, X_batch, y_batch):
    # Локальное накопление градиентов
    batch_gradients = [np.zeros_like(w) for w in model.weights]
    batch_loss = 0.0
    for x, y in zip(X_batch, y_batch):
        output, activations, pre_activations = model.forward(x)
        loss = l2_hinge_loss(output[np.newaxis, :], y[np.newaxis, :])
        batch_loss += loss
        gradients = model.backward(output, y, activations, pre_activations)
        for i in range(len(batch_gradients)):
            batch_gradients[i] += gradients[i]

    # Усреднение
    for i in range(len(batch_gradients)):
        batch_gradients[i] /= len(X_batch)

    return (batch_loss / len(X_batch), batch_gradients)


def train(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, n_jobs=4):
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Перемешиваем данные
        indices = np.random.permutation(n_samples)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        # Разбиваем на батчи
        batches = [
            (X_train_shuffled[i:i+batch_size], y_train_shuffled[i:i+batch_size])
            for i in range(0, n_samples, batch_size)
        ]

        epoch_loss = 0.0
        # Параллельная обработка батчей
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(executor.map(lambda b: train_on_batch(model, b[0], b[1]), batches), 
                                 total=len(batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"))
        
        # Суммируем потери и градиенты
        total_gradients = [np.zeros_like(w) for w in model.weights]
        for (loss_val, grads) in results:
            epoch_loss += loss_val
            for i in range(len(total_gradients)):
                total_gradients[i] += grads[i]

        # Усредняем градиенты по количеству батчей
        for i in range(len(total_gradients)):
            total_gradients[i] /= len(batches)

        # Обновляем веса
        model.update_weights(total_gradients)

        epoch_energy = calculate_epoch_energy(model, X_train_shuffled)

        test_accuracy = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(batches):.4f}, Test Accuracy: {test_accuracy:.2f}%, Avg Energy: {epoch_energy:.4f}")