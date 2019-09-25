import numpy as np


def cal_alpha_from_FB(F_pareto, B_pareto, U, F_ind, B_ind):
    F = np.array(np.squeeze(F_pareto[F_ind]))
    B = np.array(np.squeeze(B_pareto[B_ind]))

    alpha = np.sum((U - B) * (F - B), 1) / ((np.sum((F - B) ** 2, 1)) + 0.0001)

    return np.array(alpha)
