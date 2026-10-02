import numpy as np
from scene import GaussianModel
import robust_laplacian_bindings_ext as rlbe
from general_utils import compute_norm, projectPointOnPointCloud
from utils.general_utils import build_scaling_rotation


def computeNaivePointCloudGeodesicDistance(path, sourcePoint, points):
    '''
    Compute geodesic distance on point cloud using heat method provided in Geometry-central
    https://github.com/nmwsharp/geometry-central
    '''
    starting_point = projectPointOnPointCloud(sourcePoint, points)

    distances = rlbe.naivePointCloudHeatDistance(path, starting_point)

    return distances.squeeze()

def computePointCloudGeodesicDistance(path, sourcePoint, points):
    '''
    Compute geodesic distance on point cloud by first calculating the laplacian...
    Extension of the function provided in Geometry-central, which separates the laplacian computation from heat method.
    '''
    starting_point = projectPointOnPointCloud(sourcePoint, points)

    distances = rlbe.PointCloudHeatDistance(path, starting_point, points, 1e-5, 30)

    return distances.squeeze()

def computeNaiveMeshGeodesicDistance(path, meshIndex):
    '''
    Compute geodesic distance on Meshes in heat method provided in Geometry-central
    '''
    # Error exists in the original implementation (try cat0 using source index 5181)
    starting_point = meshIndex

    distances = rlbe.naiveMeshHeatDistance(path, starting_point)

    return distances.squeeze()

def computeMeshGeodesicDistance(path, meshIndex):
    '''
    Compute Exact geodesic distance on Meshes
    '''
    starting_point = meshIndex

    distances = rlbe.meshHeatDistance(path, starting_point)

    return distances.squeeze()

def computeGaussianGeodesicDistance(path, sourcePoint, gaussians):
    '''
    Compute geodesic distance on Gaussians with normals computed from covariance and neighbors computed in Euclidean distance.
    '''
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    starting_point = projectPointOnPointCloud(sourcePoint, points)
    
    norms, (covs, S) = compute_norm(gaussians)
    distances = rlbe.naiveGaussianHeatDistance(path, starting_point, points, norms, S[..., 2:3], 1e-5, 30)

    return distances.squeeze()

def computeGaussianGeodesicDistanceMahalanobis2GraphFiltration(nNeigh, sourcePoint, gaussians, index, use_normal=True):
    '''
    Compute geodesic distance on Gaussians with neighbors computed in Mahalanobis distance
    '''
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)
    assert RT_inverse.shape[0] == points.shape[0]
    
    if index is not None:
        starting_point = projectPointOnPointCloud(sourcePoint, points[index])
        distances = rlbe.naiveGaussianHeatDistanceMahalanobis2(starting_point, points[index], norms[index], RT_inverse[index], 1e-5, 100, nNeigh, use_normal)
    else:
        starting_point = projectPointOnPointCloud(sourcePoint, points)
        distances = rlbe.naiveGaussianHeatDistanceMahalanobis2(starting_point, points, norms, RT_inverse, 1e-5, 100, nNeigh, use_normal)

    return distances.squeeze()