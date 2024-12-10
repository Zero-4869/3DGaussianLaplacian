import numpy as np
from scene import GaussianModel
import robust_laplacian_bindings_ext as rlbe
from extensions.utils import compute_norm
from utils.general_utils import build_scaling_rotation


def BFS(neighs):
    '''Implement the breath first search'''
    Npts = len(neighs)
    is_search = np.zeros(Npts).astype(bool)
    cc = np.zeros(Npts, dtype=int)
    N_components = 0
    for i in range(Npts):
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

def GraphFiltration(gaussians, radiusNeigh = 80, nNeigh = 10):
    '''
    Filter the 3DGS and keep the largest connected component
    args: 
        radiusNeigh: number of neighbors pre-filtered in Euclidean distance
        nNeigh: number of uni-lateral neighbors to keep in Mahalanobis distance
    '''
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)
    assert RT_inverse.shape[0] == points.shape[0]

    neighbors= rlbe.neighborhoodMahalanobis_bilateral(points, norms, RT_inverse, 1e-5, radiusNeigh, nNeigh)
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