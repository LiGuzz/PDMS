import numpy as np
from cal_alpha_from_FB import cal_alpha_from_FB


def evaluate_samples(pareto_F, nrof_pareto_F_solution, pareto_B, nrof_pareto_B_solution, spatial_cost_UF,
                     spatial_cost_UB, F_color, B_color, U_color):
    nrof_unknown = U_color.shape[0]
    best_alpha = np.zeros(nrof_unknown)

    # each unknown pixel
    for i in range(nrof_unknown):
        # calculate alpha values
        F_pareto_dist = spatial_cost_UF[i, pareto_F[i, :nrof_pareto_F_solution[i]]]
        B_pareto_dist = spatial_cost_UB[i, pareto_B[i, :nrof_pareto_B_solution[i]]]
        F_pareto_color = F_color[pareto_F[i, :nrof_pareto_F_solution[i]]]
        B_pareto_color = B_color[pareto_B[i, :nrof_pareto_B_solution[i]]]

        F_ind = np.array([range(nrof_pareto_F_solution[i])] * nrof_pareto_B_solution[i]).reshape(-1, 1)
        B_ind = np.array([range(nrof_pareto_B_solution[i])] * nrof_pareto_F_solution[i]).T.reshape(-1, 1)

        FB_pareto_alpha = cal_alpha_from_FB(np.matrix(F_pareto_color.astype(int)),
                                            np.matrix(B_pareto_color.astype(int)),
                                            U_color[i].astype(int), F_ind, B_ind)

        # color distortion
        FB_pareto_alpha[FB_pareto_alpha > 1] = 1
        FB_pareto_alpha[FB_pareto_alpha < 0] = 0
        # I - (alpha * F + (1 - alpha) * B)
        color_distortion = U_color[i] - (FB_pareto_alpha.reshape(-1, 1) * np.array(np.matrix(F_pareto_color[F_ind]))
                                      + (1 - FB_pareto_alpha.reshape(-1, 1)) * np.array(np.matrix(B_pareto_color[B_ind])))
        color_distortion = np.exp(-(np.sqrt(np.sum(np.array(color_distortion) ** 2, 1))))

        F_dist = np.exp(-F_pareto_dist[F_ind] / np.mean(F_pareto_dist))
        B_dist = np.exp(-B_pareto_dist[B_ind] / np.mean(B_pareto_dist))

        fitness = (color_distortion ** 0.5) * (B_dist.T * F_dist.T) ** 0.5
        best_alpha[i] = FB_pareto_alpha[np.argmax(fitness)]

    return best_alpha
