import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from spin_nn.model import MSKModel
from spin_nn.temp_calc import calc_min_b

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

   # График 1: Потери
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)
    plt.title("Изменение функции потерь от эпох", fontsize=16, pad=15)
    plt.xlabel("Эпохи", fontsize=14, labelpad=10)
    plt.ylabel("Loss", fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # График 2: Точность
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='green', linewidth=2, markersize=6)
    plt.title("Изменение точности от эпох", fontsize=16, pad=15)
    plt.xlabel("Эпохи", fontsize=14, labelpad=10)
    plt.ylabel("Точность (%)", fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # График 3: Энергия
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, energies, marker='o', linestyle='-', color='red', linewidth=2, markersize=6)
    plt.title("Изменение энергии системы от эпох", fontsize=16, pad=15)
    plt.xlabel("Эпохи", fontsize=14, labelpad=10)
    plt.ylabel("Энергия", fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()




    def power_law_with_offset(x, a, alpha, c):
        return a + c * x ** alpha
    
    initial_guess = [1.0, 0.3, 1.0]

    # Аппроксимация
    params, _ = curve_fit(power_law_with_offset, epochs, curie_temperatures, p0=initial_guess, maxfev=5000)
    a, alpha, c = params
    fitted_temperatures = power_law_with_offset(epochs[5:], a, alpha, c)

    # График 4: Температура Кюри
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, curie_temperatures, marker='o', linestyle='-', color='purple', linewidth=2, markersize=6)
    #plt.plot(epochs[5:], fitted_temperatures, '--', color='orange', label=f"Аппроксимация: $t^{{{alpha:.3f}}}$")
    plt.title("$T_c$ от эпох", fontsize=16, pad=15)
    plt.xlabel("Эпохи", fontsize=14, labelpad=10)
    plt.ylabel("$T_{c}$", fontsize=14, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    print(a, alpha, c)





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
    plt.imshow(spin_grid, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
    plt.title(f"Визуализация спинов для слоя {layer_index} ({num_neurons} спинов)", fontsize=14, pad=15)
    plt.axis('off') 
    plt.tight_layout() 
    plt.show()




def visualize_Tc(weights_folder):
    """
    Строит зависимость температуры Кюри от эпохи
    и сохраняет изображения в папку `processed_data`.
    """
    # Получение базовой директории
    base_dir = os.path.dirname(weights_folder)
    folder_name = os.path.basename(weights_folder)
    
    # Путь для сохранения обработанных данных
    processed_data_dir = os.path.join(base_dir, "processed_data", folder_name)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Получение списка только JSON-файлов
    weight_files = sorted(
    (f for f in os.listdir(weights_folder) if f.startswith("weights_epoch_") and f.endswith(".json")),
    key=lambda x: int(x.split("_")[-1].split(".")[0]))
    metrics_file = os.path.join(weights_folder, "metrics.json")

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

        
    
    epochs = []
    critical_temperatures = []

    for weight_file in weight_files:
        # Извлечение номера эпохи из названия файла
        try:
            epoch = int(weight_file.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"Пропуск файла с некорректным названием: {weight_file}")
            continue

        # Полный путь до файла весов
        weight_path = os.path.join(weights_folder, weight_file)
        
        # Загрузка модели с весами
        try:
            model = MSKModel.load_weights(weight_path)
        except Exception as e:
            print(f"Ошибка при загрузке модели из файла {weight_path}: {e}")
            continue
    
        

        beta = calc_min_b(model.weights)

        epochs.append(epoch)
        critical_temperatures.append(1 / beta)  # T_c = 1 / beta

       
    sorted_indices = np.argsort(epochs)
    metrics["critical_temperatures"] = list(np.array(critical_temperatures)[sorted_indices])

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


    plt.figure(figsize=(8, 6))
    plt.plot(np.array(metrics["epochs"]), np.array(metrics["critical_temperatures"]), marker='o', linestyle='None', color='blue', label="$T_c = 1 / \\beta$")
    plt.xlabel("Эпоха", fontsize=14)
    plt.ylabel("$T_c$", fontsize=14)
    plt.title("Зависимость $T_c$ от эпохи", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

