from spin_nn.equations import total_energy_parallel
from spin_nn.model import MSKModel
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

def evaluate_model_parallel(model, X_test, y_test):
    """
    Параллельная оценка точности модели на тестовой выборке.
    """
    def compute_accuracy(x, y):
        output = model.forward(x)
        predicted_class = np.argmax(output)
        return predicted_class == y

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_accuracy, X_test, y_test))

    accuracy = sum(results) / len(results)
    return accuracy


def generate_new_weights_parallel(model):
    """
    Генерация новых весов с помощью многопоточности.
    """

    def initialize_weights(model):
        weights = [
            np.random.normal(0, 1 / np.sqrt(model.layer_sizes[i]),
                             size=(model.layer_sizes[i], model.layer_sizes[i+1]))
            for i in range(model.n_layers - 1)
        ]
        return weights

    def perturb_weight(model):
        delta_weight = initialize_weights(model)
        return model.weight + 0.5 * delta_weight

    with ThreadPoolExecutor() as executor:
        new_weights = list(executor.map(perturb_weight, model))
    return new_weights


def annealing_parallel(
    model, X_train, y_train, X_test, y_test,
    beta=1.0, target_accuracy=0.95, max_iterations=10000
):


    current_energy = total_energy_parallel(model, X_train, y_train, beta)
    best_model = model
    best_accuracy = 0

    # Создаём прогресс-бар
    with tqdm(total=max_iterations, desc="Annealing Progress", unit="iter") as pbar:
        for iteration in range(max_iterations):
            # Параллельное изменение весов
            new_weights = generate_new_weights_parallel(model)

            # Создаём временную модель с новыми весами
            temp_model = MSKModel(model.layer_sizes, beta)
            temp_model.weights = new_weights

            # Параллельная оценка энергии
            new_energy = total_energy_parallel(temp_model, X_train, y_train, beta)

            # Изменение энергии
            delta_energy = new_energy - current_energy

            # Принятие нового состояния
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
                model.weights = new_weights
                current_energy = new_energy


            # Параллельная проверка точности на тестовой выборке
            accuracy = evaluate_model_parallel(model, X_test, y_test)

            # Сохранение лучшей модели
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                pbar.write(f"New best accuracy {best_accuracy * 100:.2f}% ")

            # Обновляем прогресс-бар
            pbar.set_postfix(
                Energy=f"{current_energy:.4f}",
                Accuracy=f"{accuracy * 100:.2f}%"
            )
            pbar.update(1)

            # Завершение при достижении целевой точности
            if accuracy >= target_accuracy:
                pbar.write(f"Target accuracy {target_accuracy * 100:.2f}% reached!")
                break

    return best_model