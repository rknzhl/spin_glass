import numpy as np
from concurrent.futures import ThreadPoolExecutor

def tap_free_energy(weights, spins, beta):
    H = 0
    current_spins = spins

    for weight in weights:
        local_field = np.dot(current_spins, weight)
        H -= 0.5 * np.sum(local_field**2)
        current_spins = np.tanh(beta * local_field)

    onsager_term = -0.5 * beta * np.sum(1 - spins**2)
    return H + onsager_term


def total_energy_parallel(model, X, y, beta):
    """
    Общая энергия системы: TAP + ошибка классификации.
    Параллельно.
    """
    energy = 0;
    def compute_energy(x, label):
        logits = model.forward(x)
        tap_energy = tap_free_energy(model.weights, x, beta)
        output = np.clip(logits, 1e-8, 1 - 1e-8)
        classification_loss = -np.log(output[label])
        return tap_energy + classification_loss

    with ThreadPoolExecutor() as executor:
        energies = list(executor.map(compute_energy, X, y))
    return np.mean(energies)