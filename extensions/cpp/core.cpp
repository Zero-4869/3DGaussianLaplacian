#include "point_cloud_utilities.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "geometrycentral/pointcloud/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"
#include "geometrycentral/pointcloud/point_cloud_io.h"

#include "geometrycentral/surface/exact_geodesics.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"

#include "geometrycentralext/heat_method_distance_ext.h"
#include "geometrycentralext/point_cloud_heat_solver_ext.h"
#include "geometrycentralext/point_cloud_io_ext.h"


namespace py = pybind11;

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;


// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Parameters related to unused elements. Maybe expose these as parameters?
double laplacianReplaceVal = 1.0;
double massReplaceVal = -1e-3;

std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildMeshLaplacian(const DenseMatrix<double>& vMat, const DenseMatrix<size_t>& fMat, double mollifyFactor) {

  // First, load a simple polygon mesh
  SimplePolygonMesh simpleMesh;

  // Copy to std vector representation
  simpleMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < simpleMesh.vertexCoordinates.size(); iP++) {
    simpleMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  simpleMesh.polygons.resize(fMat.rows());
  for (size_t iF = 0; iF < simpleMesh.polygons.size(); iF++) {
    simpleMesh.polygons[iF] = std::vector<size_t>{fMat(iF, 0), fMat(iF, 1), fMat(iF, 2)};
  }

  // Remove any unused vertices
  std::vector<size_t> oldToNewMap = simpleMesh.stripUnusedVertices();


  // Build the rich mesh data structure
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  // Do the hard work, calling the geometry-central function
  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L *= 2;
  M *= 2;
  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(simpleMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildPointCloudLaplacian(const DenseMatrix<double>& vMat,
                                                                                double mollifyFactor, size_t nNeigh) {

  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }

  // Generate the local triangulations for the point cloud
  Neighbors_t neigh = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2;
  M = M * 2;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}


std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildPointCloudLaplacian_with_neigh(const DenseMatrix<double>& vMat,
                                                                                const DenseMatrix<size_t>& nbMat,
                                                                                const DenseMatrix<double>& nMat,
                                                                                double mollifyFactor) {

  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }

  // Generate the local triangulations for the point cloud
  Neighbors_t neigh;
  for(size_t i=0;i<nbMat.rows();i++){
    std::vector<size_t> this_neigh;
    for(size_t j=0;j<nbMat.cols();j++){
      if(nbMat.coeff(i, j) == nbMat.rows()){break;}
      this_neigh.emplace_back(nbMat.coeff(i, j));
    }
    neigh.emplace_back(this_neigh);
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2;
  M = M * 2;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildGaussianLaplacian(const DenseMatrix<double>& vMat, const DenseMatrix<double>& nMat, const DenseMatrix<double>& cMat, double mollifyFactor, size_t nNeigh){
  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
 
  // Generate the local triangulations for the point cloud
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  // std::vector<double> thresholds(cMat.rows());
  // for(size_t i=0;i<cMat.rows();i++){
  //   // thresholds[i] = 3 * cMat.coeff(i, 0);
  //   thresholds[i] = 0.1;
  // }
  // size_t N = 30;
  // Neighbors_t neigh = filter_knn(cloudMesh.vertexCoordinates, neigh_raw, normals, N);
  Neighbors_t neigh = neigh_raw;
  // std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  // normals = generate_normals(cloudMesh.vertexCoordinates, neigh);

  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2.;
  M = M * 2.;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

// use mahalanobis distances as filteration
std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildGaussianLaplacian_mahalanobis2(const DenseMatrix<double>& vMat, const DenseMatrix<double>& nMat, \
                                                                                            const DenseMatrix<double>& RT_inverse, double mollifyFactor, \
                                                                                            size_t nNeigh, size_t N, bool use_normal){
  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate the local triangulations for the point cloud
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  // std::vector<double> thresholds(cMat.rows());
  // for(size_t i=0;i<cMat.rows();i++){
  //   // thresholds[i] = 3 * cMat.coeff(i, 0);
  //   thresholds[i] = 0.1;
  // }
  // size_t N = 10; // 8
  Neighbors_t neigh = filter_mahalanobis(cloudMesh.vertexCoordinates, neigh_raw, N, RT_inverse);
  // Neighbors_t neigh = neigh_raw;
  // std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  // normals = generate_normals(cloudMesh.vertexCoordinates, neigh);

  // Generate normals
  
  std::vector<Vector3> normals(nMat.rows());
  if(use_normal){
    for(size_t i=0;i<nMat.rows();i++){
      normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
    }
  }
  else{
    normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  }
  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2.;
  M = M * 2.;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

// use mahalanobis distances as filteration, and add bilateral recognition
std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildGaussianLaplacian_mahalanobis_bilateral(const DenseMatrix<double>& vMat, const DenseMatrix<double>& nMat, const DenseMatrix<double>& RT_inverse, double mollifyFactor, size_t nNeigh){
  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
 
  // Generate the local triangulations for the point cloud
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  // std::vector<double> thresholds(cMat.rows());
  // for(size_t i=0;i<cMat.rows();i++){
  //   // thresholds[i] = 3 * cMat.coeff(i, 0);
  //   thresholds[i] = 0.1;
  // }
  size_t N = 10;
  Neighbors_t neigh = filter_mahalanobis_bilateral(cloudMesh.vertexCoordinates, neigh_raw, N, RT_inverse);
  // Neighbors_t neigh = neigh_raw;
  // std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  // normals = generate_normals(cloudMesh.vertexCoordinates, neigh);

  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2.;
  M = M * 2.;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildGaussianLaplacian_mahalanobis(const DenseMatrix<double>& vMat, const DenseMatrix<double>& nMat, const DenseMatrix<double>& RT_inverse, double mollifyFactor, size_t nNeigh){
  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
  // Generate the local triangulations for the point cloud
  // Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  Neighbors_t neigh_raw = generate_knn_mahalanobis_euclidean(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  // Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse, 5);

  // size_t N = 25;
  // Neighbors_t neigh = filter_knn(cloudMesh.vertexCoordinates, neigh_raw, normals, N);
  Neighbors_t neigh = neigh_raw;
  // std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);

  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);

  // Take the union of all triangles in all the neighborhoods
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);

  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor);

  L = L * 2.;
  M = M * 2.;

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

DenseMatrix<double> meshHeatDistance(std::string plyPath, size_t sourceIndex){
  // Load a mesh
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readSurfaceMesh(plyPath);

  // Pick a vertex
  Vertex sourceVert = mesh->vertex(sourceIndex); /* some vertex */

  // Compute distance
  VertexData<double> distToSource = exactGeodesicDistance(*mesh, *geometry, sourceVert);
  DenseMatrix<double> distances(mesh->nVertices(), 1);
  for(size_t iPt = 0; iPt<mesh->nVertices(); iPt++){
    distances(iPt, 0) = distToSource[mesh->vertex(iPt)];
  }
  return distances;
}

DenseMatrix<double> naiveMeshHeatDistance(std::string plyPath, size_t sourceIndex){
  // Read in a point cloud
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readSurfaceMesh(plyPath);

  // Create the solver
  HeatMethodDistanceSolver solver(*geometry, 1, true);
  // Compute geodesic distance
  Vertex pSource = mesh->vertex(sourceIndex);

  // Compute geodesic distance
  VertexData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(mesh->nVertices(), 1);
  for(size_t iPt = 0; iPt<mesh->nVertices(); iPt++){
    distances(iPt, 0) = distance[mesh->vertex(iPt)];
  }
  return distances;
}  

DenseMatrix<double> naiveMeshHeatDistance_Laplacian(std::string plyPath, 
                                                  SparseMatrix<double>& L,
                                                  SparseMatrix<double>& M,
                                                  size_t sourceIndex)
{
  // Read in a point cloud
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readSurfaceMesh(plyPath);

  // Create the solver
  HeatMethodDistanceSolverExt solver(*geometry, L, M, 1);
  // Compute geodesic distance
  Vertex pSource = mesh->vertex(sourceIndex);

  // Compute geodesic distance
  VertexData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(mesh->nVertices(), 1);
  for(size_t iPt = 0; iPt<mesh->nVertices(); iPt++){
    distances(iPt, 0) = distance[mesh->vertex(iPt)];
  }
  return distances;
} 

DenseMatrix<double> naivePointCloudHeatDistance(std::string plyPath, size_t sourceIndex){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometry> geom;
  std::tie(cloud, geom) = readPointCloud(plyPath);

  // Create the solver
  PointCloudHeatSolver solver(*cloud, *geom, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

DenseMatrix<double> PointCloudHeatDistance(std::string plyPath, size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // compute the Laplacian
  auto laplacian = buildPointCloudLaplacian(vMat, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  Neighbors_t neigh = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
 
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

DenseMatrix<double> PointCloudHeatDistance2(size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::vector<std::array<double, 3>> vPos;
  for(size_t iP = 0; iP < vMat.rows(); iP++){
    std::array<double, 3> this_pos = {vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
    vPos.emplace_back(this_pos);
  }
  std::tie(cloud, geom) = readPointCloudExt(vPos);

  // compute the Laplacian
  auto laplacian = buildPointCloudLaplacian(vMat, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  Neighbors_t neigh = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
 
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);

  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

DenseMatrix<double> naiveGaussianHeatDistance(std::string plyPath, size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          const DenseMatrix<double>& nMat, 
                                          const DenseMatrix<double>& cMat, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // compute the Laplacian
  auto laplacian = buildGaussianLaplacian(vMat, nMat, cMat, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  // size_t N = 20;
  // Neighbors_t neigh = filter_knn(cloudMesh.vertexCoordinates, neigh_raw, normals, N);
  Neighbors_t neigh = neigh_raw;
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

DenseMatrix<double> naiveGaussianHeatDistance2(size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          const DenseMatrix<double>& nMat, 
                                          const DenseMatrix<double>& cMat, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::vector<std::array<double, 3>> vPos;
  for(size_t iP = 0; iP < vMat.rows(); iP++){
    std::array<double, 3> this_pos = {vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
    vPos.emplace_back(this_pos);
  }
  std::tie(cloud, geom) = readPointCloudExt(vPos);

  // compute the Laplacian
  auto laplacian = buildGaussianLaplacian(vMat, nMat, cMat, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  std::vector<Vector3> normals(nMat.rows());
  for(size_t i=0;i<nMat.rows();i++){
    normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  }
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  // size_t N = 30;
  // Neighbors_t neigh = filter_knn(cloudMesh.vertexCoordinates, neigh_raw, normals, N);
  Neighbors_t neigh = neigh_raw;
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}


DenseMatrix<double> naiveGaussianHeatDistanceMahalanobis(std::string plyPath, size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          const DenseMatrix<double>& nMat, 
                                          const DenseMatrix<double>& RT_inverse, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // compute the Laplacian
  auto laplacian = buildGaussianLaplacian_mahalanobis(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  // auto laplacian = buildGaussianLaplacian(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  // std::vector<Vector3> normals(nMat.rows());
  // for(size_t i=0;i<nMat.rows();i++){
  //   normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  // }
  // Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  Neighbors_t neigh_raw = generate_knn_mahalanobis_euclidean(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  // Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse, 5);
  // size_t N = 25;
  // Neighbors_t neigh = filter_knn(cloudMesh.vertexCoordinates, neigh_raw, normals, N);
  Neighbors_t neigh = neigh_raw;
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    // distances(iPt, 0) = 1.;
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

// use mahalanobis distance as filtration
DenseMatrix<double> naiveGaussianHeatDistanceMahalanobis2(size_t sourceIndex, 
                                          const DenseMatrix<double>& vMat, 
                                          const DenseMatrix<double>& nMat, 
                                          const DenseMatrix<double>& RT_inverse, 
                                          double mollifyFactor, size_t nNeigh, size_t N, bool use_normal){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::vector<std::array<double, 3>> vPos;
  for(size_t iP = 0; iP < vMat.rows(); iP++){
    std::array<double, 3> this_pos = {vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
    vPos.emplace_back(this_pos);
  }
  std::tie(cloud, geom) = readPointCloudExt(vPos);

  // compute the Laplacian
  auto laplacian = buildGaussianLaplacian_mahalanobis2(vMat, nMat, RT_inverse, mollifyFactor, nNeigh, N, use_normal);
  // auto laplacian = buildGaussianLaplacian(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);

  // size_t N = 8;
  Neighbors_t neigh = filter_mahalanobis(cloudMesh.vertexCoordinates, neigh_raw, N, RT_inverse);
  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M, neigh, 1);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    // distances(iPt, 0) = 1.;
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}

DenseMatrix<double> naiveGaussianHeatDistancett(std::string plyPath, size_t sourceIndex, 
                                          SparseMatrix<double>& L,
                                          SparseMatrix<double>& M){
  // Read in a point cloud
  std::unique_ptr<PointCloud> cloud;
  std::unique_ptr<PointPositionGeometryExt> geom;
  std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // Create the solver
  PointCloudHeatSolverExt solver(*cloud, *geom, L, M);
  // Compute geodesic distance
  Point pSource = cloud->point(sourceIndex);

  // Compute geodesic distance
  PointData<double> distance = solver.computeDistance(pSource); // PointData<T> = MeshData<Point, T>
  DenseMatrix<double> distances(cloud->nPoints(), 1);
  for(size_t iPt = 0; iPt<cloud->nPoints(); iPt++){
    distances(iPt, 0) = distance[cloud->point(iPt)];
  }
  return distances;
}


std::tuple<SparseMatrix<double>, SparseMatrix<double>> naivePointCloudLaplacian(std::string plyPath){
  std::unique_ptr<PointCloud> cloud_;
  std::unique_ptr<PointPositionGeometry> geom_;
  // IntrinsicGeometryInterface&
  std::tie(cloud_, geom_) = readPointCloud(plyPath);
  PointPositionGeometry& geom(*geom_);

  geom.requireNeighbors();
  geom.requireTuftedTriangulation();
  // geom.tuftedGeom->requireEdgeLengths();
  // geom.requireTangentCoordinates();
  // geom.requireNeighbors();
    // Mass matrix
  geom.tuftedGeom->requireVertexLumpedMassMatrix();
  SparseMatrix<double>& M = geom.tuftedGeom->vertexLumpedMassMatrix;

  // Laplacian
  geom.tuftedGeom->requireCotanLaplacian();
  SparseMatrix<double>& L = geom.tuftedGeom->cotanLaplacian;
  geom.tuftedGeom->unrequireVertexLumpedMassMatrix();
  geom.tuftedGeom->unrequireCotanLaplacian();
  return std::make_tuple(L, M);
}


// visualize the neighbors of Mahalanobis distance
std::tuple<DenseMatrix<size_t>, DenseMatrix<double>> neighborhoodMahalanobis(
                                          const DenseMatrix<double>& vMat, 
                                          const DenseMatrix<double>& nMat, 
                                          const DenseMatrix<double>& RT_inverse, 
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  // std::unique_ptr<PointCloud> cloud;
  // std::unique_ptr<PointPositionGeometryExt> geom;
  // std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // // compute the Laplacian
  // // auto laplacian = buildGaussianLaplacian_mahalanobis(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  // auto laplacian = buildGaussianLaplacian(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  // SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  // std::vector<Vector3> normals(nMat.rows());
  // for(size_t i=0;i<nMat.rows();i++){
  //   normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  // }
  // Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse, 5);
  Neighbors_t neigh_raw = generate_knn_adapted(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  // Neighbors_t neigh_raw = generate_knn_mahalanobis_euclidean(cloudMesh.vertexCoordinates, nNeigh, RT_inverse);
  
  DenseMatrix<size_t> neigh(neigh_raw.size(), nNeigh);
  for(size_t iN=0; iN<neigh_raw.size(); iN++){
    for(size_t iPt=0; iPt<neigh_raw[iN].size(); iPt++){
      neigh(iN, iPt) = neigh_raw[iN][iPt];
    }
  }
  // std::vector<Vector3> normals_(nMat.rows());
  // for(size_t i=0;i<nMat.rows();i++){
  //   normals_[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  // }
  std::vector<Vector3> normals_ = generate_normals(cloudMesh.vertexCoordinates, neigh_raw);
  DenseMatrix<double> normals(normals_.size(), 3);
  for(size_t iN=0; iN<normals_.size(); iN++){
    for(size_t iPt=0; iPt<3; iPt++){
      normals(iN, iPt) = normals_[iN][iPt];
    }
  }
  return std::make_tuple(neigh, normals);
}


DenseMatrix<size_t> neighborhoodMahalanobis_bilateral(
                                      const DenseMatrix<double>& vMat, 
                                      const DenseMatrix<double>& nMat, 
                                      const DenseMatrix<double>& RT_inverse, 
                                      double mollifyFactor, size_t nNeigh, size_t N){
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }

  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);

  // size_t N = 10;
  Neighbors_t neigh_ = filter_mahalanobis_bilateral(cloudMesh.vertexCoordinates, neigh_raw, N, RT_inverse);

  DenseMatrix<size_t> neigh(neigh_.size(), N);
  for(size_t iN=0; iN<neigh_.size(); iN++){
    for(size_t iPt=0; iPt<neigh_[iN].size(); iPt++){
      neigh(iN, iPt) = neigh_[iN][iPt];
    }
    for(size_t iPt=neigh_[iN].size(); iPt<N;iPt++){
      neigh(iN, iPt) = neigh_.size();
    }
  }

  return neigh;
}


DenseMatrix<size_t> neighborhoodKNN(
                                          const DenseMatrix<double>& vMat,
                                          double mollifyFactor, size_t nNeigh){
  // Read in a point cloud
  // std::unique_ptr<PointCloud> cloud;
  // std::unique_ptr<PointPositionGeometryExt> geom;
  // std::tie(cloud, geom) = readPointCloudExt(plyPath);

  // // compute the Laplacian
  // // auto laplacian = buildGaussianLaplacian_mahalanobis(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  // auto laplacian = buildGaussianLaplacian(vMat, nMat, RT_inverse, mollifyFactor, nNeigh);
  // SparseMatrix<double> L = std::get<0>(laplacian); SparseMatrix<double> M = std::get<1>(laplacian);
  // compute the neighbors
  SimplePolygonMesh cloudMesh;

  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  // Generate normals
  // std::vector<Vector3> normals(nMat.rows());
  // for(size_t i=0;i<nMat.rows();i++){
  //   normals[i] = Vector3{nMat.coeff(i, 0), nMat.coeff(i, 1), nMat.coeff(i, 2)};
  // }
  Neighbors_t neigh_raw = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  DenseMatrix<size_t> neigh(neigh_raw.size(), nNeigh);
  for(size_t iN=0; iN<neigh_raw.size(); iN++){
    for(size_t iPt=0; iPt<neigh_raw[iN].size(); iPt++){
      neigh(iN, iPt) = neigh_raw[iN][iPt];
    }
  }
  return neigh;
}

// Actual binding code
// clang-format off
PYBIND11_MODULE(robust_laplacian_bindings_ext, m) {
  m.doc() = "Robust laplacian low-level bindings extension";


  m.def("buildMeshLaplacian", &buildMeshLaplacian, "build the mesh Laplacian", 
      py::arg("vMat"), py::arg("fMat"), py::arg("mollifyFactor"));
  m.def("buildPointCloudLaplacian", &buildPointCloudLaplacian, "build the point cloud Laplacian", 
      py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("buildPointCloudLaplacian_with_neigh", &buildPointCloudLaplacian_with_neigh, "build the point cloud Laplacian with predefined connectivity", 
      py::arg("vMat"), py::arg("nbMat"), py::arg("nMat"), py::arg("mollifyFactor"));
  m.def("buildGaussianLaplacian", &buildGaussianLaplacian, "build the 3d-Gaussian Laplacian",
      py::arg("vMat"), py::arg("nMat"), py::arg("cMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("buildGaussianLaplacian_mahalanobis", &buildGaussianLaplacian_mahalanobis, "build the 3d-Gaussian Laplacian with Mahalanobis KNN",
      py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("buildGaussianLaplacian_mahalanobis2", &buildGaussianLaplacian_mahalanobis2, "build the 3d-Gaussian Laplacian with Mahalanobis KNN",
      py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"), py::arg("N"), py::arg("use_normal"));
  m.def("buildGaussianLaplacian_mahalanobis_bilateral", &buildGaussianLaplacian_mahalanobis_bilateral, "build the 3d-Gaussian Laplacian with Mahalanobis KNN and bilateral recognition",
      py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("naivePointCloudHeatDistance", &naivePointCloudHeatDistance, "Compute geodesic distance from a source on a point cloud",
      py::arg("plyPath"), py::arg("sourceIndex"));
  m.def("naiveMeshHeatDistance_Laplacian", &naiveMeshHeatDistance_Laplacian, "Compute the geodesic distance given L and M",
      py::arg("plyPath"), py::arg("L"), py::arg("M"), py::arg("sourceIndex"));
  m.def("naiveMeshHeatDistance", &naiveMeshHeatDistance, "Compute geodesic distance from a source on a mesh in a naive way",
    py::arg("plyPath"), py::arg("sourceIndex"));
  m.def("meshHeatDistance", &meshHeatDistance, "Compute geodesic distance from a source on a mesh",
      py::arg("plyPath"), py::arg("sourceIndex"));
  m.def("naiveGaussianHeatDistance", &naiveGaussianHeatDistance, "compute geodesic distance using 3D Gaussian splatting information",
      py::arg("plyPath"), py::arg("sourceIndex"),py::arg("vMat"), py::arg("nMat"), py::arg("cMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("naiveGaussianHeatDistance2", &naiveGaussianHeatDistance2, "compute geodesic distance using 3D Gaussian splatting information",
      py::arg("sourceIndex"),py::arg("vMat"), py::arg("nMat"), py::arg("cMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("PointCloudHeatDistance", &PointCloudHeatDistance, "compute geodesic distance using point cloud information",
      py::arg("plyPath"), py::arg("sourceIndex"),py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("PointCloudHeatDistance2", &PointCloudHeatDistance2, "compute geodesic distance using point cloud information",
      py::arg("sourceIndex"),py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("naiveGaussianHeatDistanceMahalanobis", &naiveGaussianHeatDistanceMahalanobis, "compute geodesic distance using 3D Gaussian splatting information with mahalanobis distance",
      py::arg("plyPath"), py::arg("sourceIndex"),py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("naiveGaussianHeatDistanceMahalanobis2", &naiveGaussianHeatDistanceMahalanobis2, "compute geodesic distance using 3D Gaussian splatting information with mahalanobis distance as filter",
      py::arg("sourceIndex"),py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"), py::arg("N"), py::arg("use_normal"));
  m.def("naivePointCloudLaplacian", &naivePointCloudLaplacian, "directly use laplacian object",
      py::arg("plyPath"));
  m.def("naiveGaussianHeatDistancett", &naiveGaussianHeatDistancett, "tt",
      py::arg("plyPath"), py::arg("sourceIndex"), py::arg("L"), py::arg("M"));
  m.def("neighborhoodMahalanobis", &neighborhoodMahalanobis, "examine the neighborhood",
      py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"));
  m.def("neighborhoodMahalanobis_bilateral", &neighborhoodMahalanobis_bilateral, "examine the neighborhood",
      py::arg("vMat"), py::arg("nMat"), py::arg("RT_inverse"), py::arg("mollifyFactor"), py::arg("nNeigh"), py::arg("N"));
  m.def("neighborhoodKNN", &neighborhoodKNN, "examine the neighborhood KNN",
      py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
}
 

// clang-format on