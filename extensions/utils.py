import time
import numpy as np
import robust_laplacian
import polyscope as ps
from plyfile import PlyData

import scipy.sparse.linalg as sla
from numpy.linalg import svd
import robust_laplacian_bindings_ext as rlbe

def laplacian_gaussian(gaussians, n_eig, tb_writer, iteration):
    # read input
    points = gaussians.get_xyz.detach().cpu().numpy()
    
    print("Total number of points = ", points.shape[0])
    # if iteration >= 20000:
    norms, (covs, S) = compute_norm(gaussians)

    # opacity = gaussians.get_opacity.cpu().detach().numpy().astype(np.float64)
    # points, norms = filter_points_opacity_cov(points, norms, opacity, S[..., 2], threshold_cov=3.77e-5)
    # build point cloud laplacian
    L, M = robust_laplacian.point_cloud_laplacian(points, 1e-5, 30)

    # compute some eigens
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    # visualize
    for i in range(n_eig):
        tb_writer.add_scalar("Eigenvalue_pc_"+str(i), evals[i], iteration)

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

import torch
from utils.general_utils import build_rotation
def compute_norm2(gaussians):
    scaling = gaussians.get_scaling
    rotation = build_rotation(gaussians.get_rotation) # N * 3 * 3
    scaling_sorted, indices = torch.sort(scaling, dim=-1, descending=True) ## descending # N * 3
    torch.gather(rotation, dim=-1, index=indices[..., None]) # N * 3 * 1
    norm = rotation[:, -1, 0].detach().cpu().numpy()
    S = scaling_sorted.detach().cpu().numpy()
    return norm, S

def resample_per_gaussian(gaussians):
    means = gaussians.get_xyz.detach().cpu().numpy()
    cov3d = gaussians.get_covariance().cpu().detach().numpy()
    covs = np.zeros((cov3d.shape[0], 3, 3))
    covs[:, 0, 0] = cov3d[..., 0]; covs[:, 0, 1] = cov3d[..., 1]; covs[:, 0, 2] = cov3d[... ,2]
    covs[:, 1, 0] = cov3d[... ,1]; covs[:, 1, 1] = cov3d[..., 3]; covs[:, 1, 2] = cov3d[..., 4]
    covs[:, 2, 0] = cov3d[..., 2]; covs[:, 2, 1] = cov3d[..., 4]; covs[:, 2, 2] = cov3d[..., 5]
    norms, S = compute_norm2(gaussians)
    sampled_centers = []
    sampled_norms = []
    for i in range(len(means)):
        
        mean = means[i]
        cov = covs[i]
        N = 3
        deviates = np.random.multivariate_normal(np.zeros(3), cov, N)
        sampled_centers.append(np.random.multivariate_normal(mean, cov, N))
        sampled_norms.append(norms[i:i+1].repeat(N, axis=0))
    sampled_centers = np.concatenate(sampled_centers, axis=0)
    sampled_norms = np.concatenate(sampled_norms, axis=0)
    return sampled_centers, sampled_norms, S

def filter_points_opacity_cov(points, norms, opacity, eig3, threshold_op=.9, threshold_cov = 0.002):
    print("Number of raw points = ", len(points))
    new_points = points[(opacity.squeeze() > threshold_op) & (eig3.squeeze() < threshold_cov)]
    new_norms = norms[(opacity.squeeze() > threshold_op) & (eig3.squeeze() < threshold_cov)]
    print("Number of filtered points = ", len(new_points))
    return new_points, new_norms

def laplacian_gaussian2(gaussians, n_eig, tb_writer, iteration, mollify_factor=1e-5, n_neighbors=30):
    points = gaussians.get_xyz.cpu().detach().numpy()
    norms, (covs, S) = compute_norm(gaussians)
    # if iteration >= 20000:
    # opacity = gaussians.get_opacity.cpu().detach().numpy().astype(np.float64)
    # points, norms = filter_points_opacity_cov(points, norms, opacity, S[..., 2], threshold_cov=3.77e-5)
    L, M = rlbe.buildGaussianLaplacian(points, norms, S[..., 2:3], mollify_factor, n_neighbors)

     # compute some eigens
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    # visualize
    for i in range(n_eig):
        tb_writer.add_scalar("Eigenvalue_gaussian_"+str(i), evals[i], iteration)

def laplacian_gaussian_f_pc(gaussians, index, n_eig, tb_writer, iteration, mollify_factor=1e-5, n_neighbors=30):
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)

    norms, (covs, S) = compute_norm(gaussians)

    assert len(points) == len(norms)

    # compute the spectrum
    L, M = robust_laplacian.point_cloud_laplacian(points[index], 1e-5, 30)

    # compute some eigens
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    # visualize
    for i in range(n_eig):
        tb_writer.add_scalar("Eigenvalue_ours(KNN)_"+str(i), evals[i], iteration)

def laplacian_gaussian_f_g(gaussians, index, n_eig, tb_writer, iteration, mollify_factor=1e-5, n_neighbors=30):
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)

    norms, (covs, S) = compute_norm(gaussians)

    assert len(points) == len(norms)

    # compute the spectrum
    L, M = rlbe.buildGaussianLaplacian(points[index], norms[index], S[index][..., 2:3], 1e-5, 30)

    # compute some eigens
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    # visualize
    for i in range(n_eig):
        tb_writer.add_scalar("Eigenvalue_ours(normal)_"+str(i), evals[i], iteration)