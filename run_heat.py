import os
import argparse
import pymesh
import numpy as np
import pandas as pd
import polyscope as ps
from tqdm import tqdm
from plyfile import PlyData
from scene import GaussianModel
import robust_laplacian_bindings_ext as rlbe
from utils_laplacian.general_utils import build_scaling_rotation
from extensions.utils import compute_norm

from utils_laplacian.heat_utils import (
    computeNaivePointCloudGeodesicDistance, 
    computePointCloudGeodesicDistance, 
    computeNaiveMeshGeodesicDistance, 
    computeMeshGeodesicDistance, 
    computeGaussianGeodesicDistance, 
    computeGaussianGeodesicDistanceMahalanobis2GraphFiltration
)
from utils_laplacian.graph_utils import GraphFiltrationPartial
from utils_laplacian.metric_utils import average_error_scalar_field

def computeGeodesicDistance(args, sourcePoint, sourceIndex, points, gaussians, index, type):
    if type == "gt":
        distances = computeMeshGeodesicDistance(args, sourceIndex)
    elif type == "pc":
        distances = computePointCloudGeodesicDistance(args, sourcePoint, points)
    elif type == "euclid":
        distances = computeGaussianGeodesicDistance(args, sourcePoint, gaussians)
    elif type == "mah":
        distances = computeGaussianGeodesicDistanceMahalanobis2GraphFiltration(args, sourcePoint, gaussians, index, True)
    else:
        raise Exception("Input type does not fit")
    return distances

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="/home/hongyuzhou/Projects/gaussian-opacity-fields/exp_nerf_synthetic/release/cat0/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--gt", type=str, default="/home/hongyuzhou/Datasets/blend_files/cat0.ply")
    parser.add_argument("--mesh", type=str, default="/home/hongyuzhou/Projects/gaussian-opacity-fields/exp_nerf_synthetic/release/cat0/test/ours_30000/fusion/mesh_binary_search_7.ply") # Mesh reconstructed from GS
    argument = parser.parse_args()
    return argument

def main():
    args = argparser()
    # load GS
    path = args.path 
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(path)
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float32)
    
    # load GT mesh
    meshPath = args.gt
    meshdata = pymesh.load_mesh(meshPath)
    mesh_points = meshdata.vertices
    sourceIndice = np.random.randint(0, len(mesh_points), 1)[0]

    # optional, to get a cleaner GS in geometry
    alphas = gaussians.get_opacity.cpu().detach().numpy().astype(np.float64).squeeze()
    index = GraphFiltrationPartial(gaussians, np.where(alphas >= 0.5)[0])

    sourcePoint = mesh_points[sourceIndice]
    distances_gt = computeMeshGeodesicDistance(args, sourceIndice)#computeNaiveMeshGeodesicDistance(args, sourceIndex)
    distances_m = computeGeodesicDistance(args, sourcePoint=sourcePoint, sourceIndex=None, points=points, gaussians=gaussians, index=index, type="mah")

    average_error_m, errors_m = average_error_scalar_field(distances_gt, mesh_points, distances_m[index], points[index], False)
    print("average_error_m = ", average_error_m)

if __name__ == "__main__":
    main()
