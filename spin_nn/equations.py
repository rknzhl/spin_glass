import numpy as np

def tap_equations(weights, magnetizations, beta):
    new_magnetizations = []
    for i in range(len(weights)):
        local_field = beta * weights[i].dot(magnetizations[i])
        onsager_correction = beta**2 * np.sum(weights[i]**2, axis=1) * (1 - magnetizations[i]**2)
        new_magnetizations.append(np.tanh(local_field - onsager_correction))
    return new_magnetizations