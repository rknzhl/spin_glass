import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import cv2

def calculate_energy(spins, weights):
    energy = 0
    n = len(spins)
    for i in range(n):
        for j in range(n):
            energy += spins[i, j] * (spins[(i+1)%n, j] + spins[i, (j+1)%n])
    return -energy

def metropolis_step(spins, weights, temperature):
    n = len(spins)
    i, j = np.random.randint(0, n, 2)
    current_spin = spins[i, j]
    flipped_spin = -current_spin

    initial_energy = calculate_energy(spins, weights)

    spins[i, j] = flipped_spin
    trial_energy = calculate_energy(spins, weights)

    delta_energy = trial_energy - initial_energy

    if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
        return spins
    else:
        spins[i, j] = current_spin
        return spins

def simulated_annealing(n, initial_temperature, final_temperature, steps_per_temperature):
    spins = np.random.choice([-1, 1], size=(n, n))
    weights = np.random.rand(5)

    temperatures = np.linspace(initial_temperature, final_temperature, steps_per_temperature)
    save_indices = np.linspace(0, steps_per_temperature - 1, 40, dtype=int)
    saved_spins = []

    # Создаем папку для сохранения изображений
    output_folder = "spin_configurations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, temperature in enumerate(temperatures):
        for _ in range(steps_per_temperature):
            for _ in range(50):
                spins = metropolis_step(spins, weights, temperature)
        if idx in save_indices:
            saved_spins.append(spins.copy())

    saved_spins = np.array(saved_spins)
    display_data = (saved_spins + 1) // 2

    cmap = ListedColormap(['blue', 'red'])
    for i, temp_idx in enumerate(save_indices):
        temp = temperatures[temp_idx]
        data = display_data[i]
        plt.figure(figsize=(4,4))
        plt.imshow(data, cmap=cmap, vmin=0, vmax=1)
        plt.title(f"T={temp:.2f}")
        plt.axis('off')
        filename = f"spin_config_T_{temp:.2f}.png"
        # Сохраняем изображение в папку
        plt.savefig(os.path.join(output_folder, filename), dpi=300)
        plt.close()

    # Создаем видео из изображений
    create_video_from_images(output_folder)

    return spins

def create_video_from_images(image_folder, output_video="output_video.mp4", fps=5):
    """
    Создает видео из изображений в указанной папке.
    :param image_folder: Папка с изображениями.
    :param output_video: Имя выходного видеофайла.
    :param fps: Количество кадров в секунду.
    """
    # Получаем список файлов в папке
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: float(x.split("_T_")[1].split(".png")[0]))  # Сортируем по температуре

    # Проверяем, есть ли изображения
    if not images:
        print("Папка не содержит изображений.")
        return

    # Загружаем первое изображение, чтобы определить размеры
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Создаем объект VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Добавляем каждое изображение в видео
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Закрываем видеофайл
    video.release()
    print(f"Видео сохранено как {output_video}")