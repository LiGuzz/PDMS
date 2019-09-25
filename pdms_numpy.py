import cv2
import numpy as np

from evaluate_samples import evaluate_samples
from tqdm import tqdm
import sys

assert len(sys.argv) ==4 or len(sys.argv) ==3 or len(sys.argv) ==1

raw_file_name = 'GT13.png'
trimap_file_name = 'trimap.png'
split_step = 10000

if len(sys.argv) >=3:
    raw_file_name = sys.argv[1]
    trimap_file_name = sys.argv[2]
if len(sys.argv) >= 4:
    split_step = int(sys.argv[3])

'''
# pytorch version
def euclidean_dist(x, y):
    m, n = np.shape(x)[0], np.shape(y)[0]
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
'''
# numpy version
def euclidean_dist(A, B):
    ABT = np.dot(A,B.T)
    AA =  A**2
    sumAA= np.expand_dims(np.sum(AA, axis=1), axis = 0)
    sumAATile = np.tile(sumAA.T, (1, ABT.shape[1]))
    BB = B**2
    sumBB = np.sum(BB, axis=1)
    sumBBTile = np.tile(sumBB, (ABT.shape[0], 1))
    ED_sq = sumBBTile + sumAATile - 2*ABT
    ED_sq[ED_sq<0]=0.0
    ED = np.sqrt(ED_sq)
    return ED

def dominate(spatial_cost_one_point, color_cost_one_point, x, y):
    diff_spatial = spatial_cost_one_point[y] - spatial_cost_one_point[x]
    diff_color = color_cost_one_point[y] - color_cost_one_point[x]
    if diff_spatial >= 0 and diff_color >= 0 and (diff_spatial > 0)|(diff_color > 0):
        return True
    else:
        return False

def swap(C, x, y):
    temp = C[x]
    C[x] = C[y]
    C[y] = temp

# Fast Discrete Multiobjective Optimization Algorithm
def FDMO(spatial_cost, color_cost):
    assert spatial_cost.shape == color_cost.shape
    nrof_unknown = spatial_cost.shape[0]
    pareto_Bw = []
    nrof_pareto_solution = []
    for n in range(nrof_unknown):
        sc = spatial_cost[n]
        cc = color_cost[n]
        C = np.arange(spatial_cost.shape[1]).tolist()
        i = 0
        j = len(C)-1
        while i <= j:
            cmp = i+1
            while cmp <= j:
                if dominate(sc, cc, C[i], C[cmp]):
                    swap(C, cmp, j)
                    j = j - 1
                elif dominate(sc, cc, C[cmp], C[i]):
                    swap(C, i, cmp)
                    swap(C, cmp, j)
                    j = j - 1
                    cmp = i + 1
                else:
                    cmp = cmp + 1
            i = i + 1
        pareto_Bw.append(C)
        nrof_pareto_solution.append(i)
    return np.array(pareto_Bw), nrof_pareto_solution

raw_img = cv2.imread(raw_file_name)
trimap = cv2.imread(trimap_file_name,0)
alpha = trimap.copy()


U_corrdinate_union = np.array_split(np.array(np.where(trimap == 128)), split_step, axis = 1)
F_corrdinate = np.array(np.where(trimap == 255))
B_corrdinate = np.array(np.where(trimap == 0))
F_color = raw_img[F_corrdinate[0],F_corrdinate[1]]
B_color = raw_img[B_corrdinate[0],B_corrdinate[1]]

for U_corrdinate in tqdm(U_corrdinate_union):
    U_color = raw_img[U_corrdinate[0],U_corrdinate[1]]
  
    spatial_cost_UF = euclidean_dist(U_corrdinate.T, F_corrdinate.T) # [step * nrof_front]
    color_cost_UF = euclidean_dist(U_color, F_color)
    pareto_F_Bw, nrof_pareto_F_solution = FDMO(spatial_cost_UF, color_cost_UF) # [step * nrof_front] , nrof_pareto_solution list

    spatial_cost_UB = euclidean_dist(U_corrdinate.T, B_corrdinate.T) # [step]
    color_cost_UB = euclidean_dist(U_color, B_color)
    pareto_B_Bw, nrof_pareto_B_solution = FDMO(spatial_cost_UB, color_cost_UB) # [step * nrof_Background] , nrof_pareto_solution list

    U_alpha = evaluate_samples(pareto_F_Bw, nrof_pareto_F_solution, pareto_B_Bw, nrof_pareto_B_solution, spatial_cost_UF,
                           spatial_cost_UB, F_color, B_color, U_color)

    alpha[U_corrdinate[0], U_corrdinate[1]] = U_alpha*255

cv2.imwrite('matte_' + raw_file_name, alpha)
print("matting finish!")
