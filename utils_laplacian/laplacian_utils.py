import numpy as np
import robust_laplacian
import scipy.sparse.linalg as sla
import robust_laplacian_bindings_ext as rlbe
from extensions.utils import compute_norm
from utils.general_utils import build_scaling_rotation

def Laplacian_mesh(meshdata, N_eigs=100):
    '''Compute mesh laplacian'''
    L, M = robust_laplacian.mesh_laplacian(meshdata.vertices, meshdata.faces)

    # compute some eigens
    n_eig = N_eigs
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    return evals, evecs, M

def Laplacian_point_cloud(points, N=30, N_eigs=100):
    '''Compute point cloud laplacian'''
    L, M = robust_laplacian.point_cloud_laplacian(points, 1e-5, N)
    n_eig = N_eigs
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    return evals, evecs, M

def Laplacian_gaussian(gaussians, index, N=30, N_eigs=100):
    '''
    compute Gaussian laplacian by taking its center points as point cloud,
    and then compute the point cloud laplacian with pre-computed normals,
    using Euclidean distance to determine the neighbors
    '''
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)

    norms, (covs, S) = compute_norm(gaussians)

    assert len(points) == len(norms)

    # compute the spectrum
    L, M = rlbe.buildGaussianLaplacian(points[index], norms[index], S[index][..., 2:3], 1e-5, N)

    # compute some eigens
    n_eig = N_eigs
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    return evals, evecs, M

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

def Laplacian_gaussian_mahalanobis(gaussians, index, N=30, use_normal = True, N_eigs = 100):
    '''
    Compute Gaussian laplacians using pre-computed normals, 
    using Mahalanobis distances to determine the neighbors.
    args: 
        index: index of Gaussians to keep after filtration
        N: number of neighbors to take before Delaunay triangulation
        use_normal: whether to use normals computed from the covariance matrix
    '''
    # read input
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
    # norms, S = compute_norm2(gaussians)
  
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    assert RT_inverse.shape[0] == points.shape[0]

    L, M = rlbe.buildGaussianLaplacian_mahalanobis2(points[index], norms[index], RT_inverse[index], 1e-5, 80, N, use_normal)
    # compute some eigens
    n_eig = N_eigs
    evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

    return evals, evecs, M

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