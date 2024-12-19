import numpy as np

def calculate_partial_hamiltonian(spins, weights, layer_idx, fixed_spins):
    """
    Вычисляет частичный вклад в энергию для спина на указанном слое.
    spins: список массивов спинов для каждого слоя (включая фиксированные).
    weights: список матриц весов между слоями.
    layer_idx: индекс слоя, для которого пересчитывается энергия.
    fixed_spins: фиксированные спины (начальный слой).
    """
    # Начальная энергия: взаимодействие с предыдущим слоем
    if layer_idx == 0:
        raise ValueError("Начальные спины фиксированы и не должны пересчитываться.")

    prev_spins = spins[layer_idx - 1]
    current_spins = spins[layer_idx]
    weight_matrix = weights[layer_idx - 1]

    # Вклад от взаимодействия с предыдущим слоем
    local_field = np.dot(prev_spins, weight_matrix)
    energy = -np.sum(local_field * current_spins)

    # Если есть следующий слой, учитываем взаимодействие с ним
    if layer_idx < len(weights):
        next_weight_matrix = weights[layer_idx]
        next_spins = spins[layer_idx + 1]
        local_field_next = np.dot(current_spins, next_weight_matrix)
        energy -= np.sum(local_field_next * next_spins)

    return energy


def metropolis_step(spins, weights, temperature, fixed_spins):
    """
    Один шаг алгоритма Метрополиса.
    spins: список массивов спинов для всех слоев.
    weights: список матриц весов между слоями.
    temperature: текущая температура.
    fixed_spins: массив фиксированных спинов.
    """
    # Выбираем случайный слой (кроме фиксированного) и случайный спин в нем
    layer_idx = np.random.randint(1, len(spins))  # Слои начиная с 1 (0 фиксирован)
    spin_idx = np.random.randint(len(spins[layer_idx]))

    current_spin = spins[layer_idx][spin_idx]
    flipped_spin = -current_spin

    # Частичная энергия до переворота
    initial_energy = calculate_partial_hamiltonian(spins, weights, layer_idx, fixed_spins)

    # Пробуем перевернуть спин
    spins[layer_idx][spin_idx] = flipped_spin

    # Частичная энергия после переворота
    trial_energy = calculate_partial_hamiltonian(spins, weights, layer_idx, fixed_spins)

    # Разница в энергии
    delta_energy = trial_energy - initial_energy

    # Критерий Метрополиса
    if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
        return spins  # Принимаем новое состояние
    else:
        # Отклоняем изменение и возвращаем спин обратно
        spins[layer_idx][spin_idx] = current_spin
        return spins


def simulated_annealing(fixed_spins, weights, initial_temperature, final_temperature, steps_per_temperature):
    """
    Метод отжига для минимизации энергии.
    fixed_spins: массив фиксированных спинов (первый слой).
    weights: список матриц весов между слоями.
    initial_temperature: начальная температура.
    final_temperature: конечная температура.
    steps_per_temperature: число шагов на каждой температуре.
    """
    # Генерируем начальные случайные спины для остальных слоев
    spins = [fixed_spins]
    for weight in weights:
        layer_size = weight.shape[1]
        spins.append(np.random.choice([-1, 1], size=layer_size))

    # Температурное расписание: линейное снижение
    temperatures = np.linspace(initial_temperature, final_temperature, steps_per_temperature)

    # Основной цикл отжига
    for temperature in temperatures:
        for _ in range(steps_per_temperature):  # Один полный цикл Монте-Карло
            spins = metropolis_step(spins, weights, temperature, fixed_spins)

    return spins
