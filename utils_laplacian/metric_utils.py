import numpy as np
from tqdm import tqdm

# compare two scalar fields on point clouds
def average_error_scalar_field(x1, pc1, x2, pc2, corr_ind = []):
    assert (len(x1.shape) == 1) & (len(x2.shape) == 1)
    # compute the nearest point index on PC2 for each point on PC1
    if len(corr_ind) == 0:
        corr_ind = []
        N1 = pc1.shape[0]; N2 = pc2.shape[0]
        N = np.minimum(20000 * 40000 // N2 // 32, N1)
        for i in range(0, N1, N):
            pc1_ = np.repeat(np.expand_dims(pc1[i:np.minimum(i+N, N1)].astype(np.float32), axis=1), N2, axis=1)
            pc2_ = np.repeat(np.expand_dims(pc2.astype(np.float32), axis=0), np.minimum(i+N, N1)-i, axis=0)
            dist = np.linalg.norm(pc1_ - pc2_, axis=2)
            corr_ind_tt = np.argmin(dist, axis=1) # (N1, )
            corr_ind.append(corr_ind_tt)
        corr_ind = np.concatenate(corr_ind)

    x1_ = x1
    x2_ = x2[corr_ind]
    errors = np.abs(x1_ - x2_)
    return np.mean(errors), x1_ - x2_, corr_ind