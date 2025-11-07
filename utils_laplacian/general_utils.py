import numpy as np

import torch
from utils.general_utils import build_rotation

def projectPointOnPointCloud(point, pc):
    dist = np.linalg.norm(pc - point, axis=-1)
    return np.argmin(dist)

def projectPointCloudOnPointCloud(points_source, points_target):
    '''For every point in points_source, find the cloest point in points_target'''
    corr_ind = []
    N1 = points_source.shape[0]; N2 = points_target.shape[0]
    N = np.minimum(20000 * 40000 // N2 // 64, N1)
    for i in range(0, N1, N):
        pc1_ = np.repeat(np.expand_dims(points_source[i:np.minimum(i+N, N1)].astype(np.float32), axis=1), N2, axis=1)
        pc2_ = np.repeat(np.expand_dims(points_target.astype(np.float32), axis=0), np.minimum(i+N, N1)-i, axis=0)
        dist = np.linalg.norm(pc1_ - pc2_, axis=2)
        corr_ind_tt = np.argmin(dist, axis=1) # (N1, )
        corr_ind.append(corr_ind_tt)
    corr_ind = np.concatenate(corr_ind)
    return corr_ind

def compute_norm(gaussians):
    start_time = time.time()
    cov3d = gaussians.get_covariance().cpu().detach().numpy()
    cov = np.zeros((cov3d.shape[0], 3, 3))
    cov[:, 0, 0] = cov3d[..., 0]; cov[:, 0, 1] = cov3d[..., 1]; cov[:, 0, 2] = cov3d[... ,2]
    cov[:, 1, 0] = cov3d[... ,1]; cov[:, 1, 1] = cov3d[..., 3]; cov[:, 1, 2] = cov3d[..., 4]
    cov[:, 2, 0] = cov3d[..., 2]; cov[:, 2, 1] = cov3d[..., 4]; cov[:, 2, 2] = cov3d[..., 5]
    U, S, V = svd(cov, compute_uv=True, hermitian=True) 
    norm = U[..., 2]
    print("elapsed time = %f s"%(time.time() - start_time))
    return norm, (cov, S)

def compute_norm2(gaussians):
    scaling = gaussians.get_scaling
    rotation = build_rotation(gaussians.get_rotation) # N * 3 * 3
    scaling_sorted, indices = torch.sort(scaling, dim=-1, descending=True) ## descending # N * 3
    torch.gather(rotation, dim=-1, index=indices[..., None]) # N * 3 * 1
    norm = rotation[:, -1, 0].detach().cpu().numpy()
    S = scaling_sorted.detach().cpu().numpy()
    return norm, S