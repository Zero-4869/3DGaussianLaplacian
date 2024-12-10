import numpy as np

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