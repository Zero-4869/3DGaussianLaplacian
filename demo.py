import os
import argparse
import numpy as np
import pandas as pd
import polyscope as ps
from tqdm import tqdm
from scene import GaussianModel
import robust_laplacian_bindings_ext as rlbe
from utils.general_utils import build_scaling_rotation
from utile_laplacian.laplacian_utils import compute_norm
import robust_laplacian

def BFS(neighs):
    Npts = len(neighs)
    is_search = np.zeros(Npts).astype(bool)
    cc = np.zeros(Npts, dtype=int)
    N_components = 0
    for i in tqdm(range(Npts)):
        if is_search[i]:
            continue
        this_component = [i]
        while len(this_component) > 0:
            source = this_component[0]
            this_component = this_component[1:]
            is_search[source] = True
            cc[source] = N_components
            for neigh in neighs[source]:
                if (not is_search[neigh]) and (not neigh in this_component):
                    this_component.append(neigh)
        N_components += 1
    return cc

def GraphFiltration(gaussians):
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)
    assert RT_inverse.shape[0] == points.shape[0]

    neighbors= rlbe.neighborhoodMahalanobis_bilateral(points, norms, RT_inverse, 1e-5, 80, 10)
    Npts = neighbors.shape[0]
    neighs = []
    for i in range(Npts):
        if neighbors[i][-1] == Npts:
            N_neighbor = np.where(neighbors[i] == Npts)[0][0]
            neighs.append(neighbors[i][:N_neighbor])
        else:
            neighs.append(neighbors[i])
    cc = BFS(neighs)
    maximum_component = 0
    N_elements = 0
    for i in range(np.max(cc)):
        if np.sum(cc==i) > N_elements:
            N_elements = np.sum(cc==i)
            maximum_component = i
    index = np.where(cc == maximum_component)[0]
    return index

def GraphFiltrationPartial(gaussians, index):
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)[index]
    norms, (covs, S) = compute_norm(gaussians)
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()[index]
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)
    assert RT_inverse.shape[0] == points.shape[0]

    neighbors= rlbe.neighborhoodMahalanobis_bilateral(points, norms[index], RT_inverse, 1e-5, 50, 10)
    Npts = neighbors.shape[0]
    neighs = []
    for i in range(Npts):
        if neighbors[i][-1] == Npts:
            N_neighbor = np.where(neighbors[i] == Npts)[0][0]
            neighs.append(neighbors[i][:N_neighbor])
        else:
            neighs.append(neighbors[i])
    cc = BFS(neighs)
    N_components = np.zeros(np.max(cc)+1)
    for i in tqdm(range(Npts)):
        N_components[cc[i]] += 1
    ordered_ids = np.argsort(np.array(N_components))
    final_index = np.where((cc == ordered_ids[-1]))[0]
    return index[final_index]

def Laplacian_point_cloud(points):
    L, M = robust_laplacian.point_cloud_laplacian(points, 1e-5, 30)

    return L, M

def Laplacian_gaussian_mahalanobis2(gaussians):
    # read input
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
  
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    assert RT_inverse.shape[0] == points.shape[0]

    L, M = rlbe.buildGaussianLaplacian_mahalanobis2(points, norms, RT_inverse, 1e-5, 100, 30, True)
    # n_eig = 100
    # evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)
    return L, M

def compute_mean_curvature(L, M, points):
    normals_weighted = L @ points
    h_weighted = np.linalg.norm(normals_weighted, axis=1)
    normals = normals_weighted / h_weighted[:, None]
    weights = M.diagonal()
    h = h_weighted / weights / (-2)
    return np.abs(h), normals

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

def Laplacian_gaussian_mahalanobis_partial(gaussians, index):

    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)[index]
    norms, (covs, S) = compute_norm(gaussians)
    norms = norms[index]
  
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()[index]
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)
    assert RT_inverse.shape[0] == points.shape[0]
    L, M = rlbe.buildGaussianLaplacian_mahalanobis2(points, norms, RT_inverse, 1e-5, 100, 30, True)
    return L, M


def argparser_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the ply file")
    return parser.parse_args()

def main(args):
    path_at = args.path 

    gaussians_at = GaussianModel(sh_degree=3)
    gaussians_at.load_ply(path_at)

    points_at = gaussians_at.get_xyz.cpu().detach().numpy().astype(np.float32)

    index_at = gaussians_at.get_opacity.cpu().detach().numpy().astype(np.float64).squeeze() #> 0.5

    L_m, M_m = Laplacian_gaussian_mahalanobis_partial(gaussians_at, index_at)
    h_m, _ = compute_mean_curvature(L_m, M_m, points_at[index_at])

    L_pc, M_pc = Laplacian_point_cloud(points_at[index_at])
    h_pc, _ = compute_mean_curvature(L_pc, M_pc, points_at[index_at])

    ps.init()
    pc_at = ps.register_point_cloud("pc_at", points_at[index_at], radius=0.003)
    pc_at.add_scalar_quantity("mean curvature Mahalanobis", h_m, enabled=True)
    pc_at.add_scalar_quantity("mean curvature Euclid", h_pc, enabled=True)
    ps.show()

if __name__ == "__main__":
    args = argparser_base()
    main(args)