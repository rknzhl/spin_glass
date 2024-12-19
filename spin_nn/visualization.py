import os
import json
import numpy as np
import matplotlib.pyplot as plt
from spin_nn.model import MSKModel

def load_metrics(json_path):
    with open(json_path, "r") as f:
        metrics = json.load(f)
    return metrics

def plot_metrics(metrics):
    epochs = metrics["epochs"]
    losses = metrics["losses"]
    accuracies = metrics["accuracies"]
    energies = metrics["energies"]
    curie_temperatures = metrics["curie_temperatures"]

    # График 1: Потери (Loss)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # График 2: Точность (Accuracy)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='green')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.show()

    # График 3: Энергия (Energy)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, energies, marker='o', linestyle='-', color='red')
    plt.title("Energy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Energy")
    plt.grid()
    plt.show()

    # График 4: Температура Кюри (Curie Temperature)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, curie_temperatures, marker='o', linestyle='-', color='purple')
    plt.title("Curie Temperature over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Curie Temperature")
    plt.grid()
    plt.show()

    # График 5: Сравнение Loss и Energy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', label="Loss", color='blue')
    plt.plot(epochs, energies, marker='x', linestyle='--', label="Energy", color='red')
    plt.title("Loss and Energy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.show()





def visualize_layer_weights(weights_folder, layer_index, step=10):
    """
    Создает визуализацию весов указанного слоя каждые step эпох 
    и сохраняет изображения в папку `processed_data`.
    """
    # Получение базовой директории
    base_dir = os.path.dirname(weights_folder)
    folder_name = os.path.basename(weights_folder)
    
    # Путь для сохранения обработанных данных
    processed_data_dir = os.path.join(base_dir, "processed_data", folder_name)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Получение списка только JSON-файлов
    weight_files = sorted(f for f in os.listdir(weights_folder) if f.startswith("weights_epoch_") and f.endswith(".json"))
    
    for weight_file in weight_files:
        # Извлечение номера эпохи из названия файла
        try:
            epoch = int(weight_file.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"Пропуск файла с некорректным названием: {weight_file}")
            continue

        # Фильтруем только эпохи кратные шагу
        if epoch % step != 0:
            continue
        
        # Полный путь до файла весов
        weight_path = os.path.join(weights_folder, weight_file)
        
        # Загрузка модели с весами
        try:
            model = MSKModel.load_weights(weight_path)
        except Exception as e:
            print(f"Ошибка при загрузке модели из файла {weight_path}: {e}")
            continue
        
        # Проверка, существует ли указанный слой
        if layer_index >= len(model.weights):
            print(f"Слой {layer_index} не найден в модели из файла {weight_file}. Пропуск.")
            continue
        
        # Извлечение весов слоя
        layer_weights = model.weights[layer_index]
        
        # Визуализация
        plt.figure(figsize=(6, 6))
        plt.imshow(layer_weights, cmap='seismic', interpolation='nearest')
        plt.colorbar(label="Вес")
        plt.title(f"Эпоха {epoch}, Слой {layer_index}")
        
        # Сохранение изображения
        save_path = os.path.join(processed_data_dir, f"epoch_{epoch}_layer_{layer_index}.png")
        plt.savefig(save_path)
        plt.close()
    
    print(f"Сохраненные изображения находятся в папке: {processed_data_dir}")



def visualize_spins_from_model(model, test_image, layer_index):
    """
    Визуализирует "спины" (красный — вверх, синий — вниз) для указанного слоя
    на основе активаций модели для тестового вектора MNIST.
    """

    if test_image.shape != (784,):
        raise ValueError("Тестовое изображение должно быть вектором длиной 784 (размер 28x28 в развернутом виде).")
    

    _, activations, _ = model.forward(test_image)
    

    if layer_index >= len(activations):
        raise ValueError(f"Слой {layer_index} не существует. У модели {len(activations)} слоев.")
    

    spin_map = activations[layer_index]

    num_neurons = spin_map.size
    grid_size = int(np.ceil(np.sqrt(num_neurons)))  # Размер сетки для отображения
    spin_grid = np.full((grid_size, grid_size), np.nan)  # Заполняем NaN для пустых ячеек
    spin_grid.flat[:num_neurons] = spin_map  # Заполняем ячейки активациями
    
    # Визуализация спинов
    plt.figure(figsize=(6, 6))
    plt.imshow(spin_grid, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label="Состояние спина")
    plt.title(f"Визуализация спинов для слоя {layer_index} ({num_neurons} спинов)")
    plt.show()