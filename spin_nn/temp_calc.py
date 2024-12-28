from spin_nn.model import MSKModel
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import numpy as np

def build_j_matrix(weights_list):
    layer_sizes = []
    # Предполагается, что weights_list не пуст
    N0 = weights_list[0].shape[0]
    layer_sizes.append(N0)
    for W in weights_list:
        out_size = W.shape[1]
        layer_sizes.append(out_size)
    total_size = sum(layer_sizes)
    J = np.zeros((total_size, total_size))

    offsets = np.cumsum([0] + layer_sizes)

    for k, W in enumerate(weights_list):
        in_start = offsets[k]
        in_end = offsets[k+1]
        out_start = offsets[k+1]
        out_end = offsets[k+2]
        # Заполняем блоки:
        J[in_start:in_end, out_start:out_end] = W
        J[out_start:out_end, in_start:in_end] = W.T
    return J



def build_M_matrix(J, beta):
    N = J.shape[0]
    # Вычисляем a_i = sum_{ell} J_{iell}^2 для каждой строки
    a = np.sum(J**2, axis=1)  # вектор длины N
    # Формируем M
    # M = beta * J - beta^2 * diag(a)
    M = beta * J - np.diag(beta**2 * a)
    return M

def find_min_eigval(J, beta):
    N = J.shape[0]
    M = build_M_matrix(J, beta)
    A = np.eye(N) - M  # I_N - M
    #eigvals = np.linalg.eigvalsh(A)  # для симметричной матрицы
    A_sparse = csr_matrix(A)
    # Находим минимальное собственное значение с помощью eigsh
    vals, _ = eigsh(A_sparse, k=1, which='SM')
    return vals[0]

def calc_min_b(weights):
    J = build_j_matrix(weights)
    answer = 10000000;
    min_eigs = []
    betas = np.linspace(0.3, 1.45, 450)
    for b in betas:
        val = find_min_eigval(J, b)
        min_eigs.append((val, b))
    for val, b in min_eigs:
        if val <= 0:
            answer = b;
            print("ПЕРЕСЕЧЕНИЕ С НУЛЕМ В:",answer)
            return answer;

    closest_val, closest_b = min(min_eigs, key=lambda x: abs(x[0]))
    print("НУЛЯ НЕТ, МИНИМУМ В:",closest_b, "   CО ЗНАЧЕНИЕМ",closest_val)
    return closest_b






