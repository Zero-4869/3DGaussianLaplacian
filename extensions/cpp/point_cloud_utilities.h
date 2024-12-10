#pragma once

#include "geometrycentral/utilities/vector2.h"
#include "geometrycentral/utilities/vector3.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"

using geometrycentral::SparseMatrix;
using geometrycentral::Vector;
using geometrycentral::Vector2;
using geometrycentral::Vector3;

// === Basic utility methods

using Neighbors_t = std::vector<std::vector<size_t>>;

// Generate the k-nearest-neighbors for the points.
// The list will _not_ include the center point.
Neighbors_t generate_knn(const std::vector<Vector3>& points, size_t k);

Neighbors_t generate_knn_adapted(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse);

Neighbors_t generate_knn_adapted(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse, double radius);

Neighbors_t generate_knn_mahalanobis_euclidean(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse);

Neighbors_t filter_mahalanobis(const std::vector<Vector3>& points, const Neighbors_t& pointInd, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse);
// std::vector<Vector3> filter_points(const std::vector<Vector3>& points, const std::vector<double>& opacity);

Neighbors_t filter_mahalanobis_bilateral(const std::vector<Vector3>& points, const Neighbors_t& pointInd, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse);

Neighbors_t filter_knn(const std::vector<Vector3>& points, Neighbors_t& neigh, const std::vector<Vector3> normals, const size_t N);

// Estimate normals from a neighborhood (arbitrarily oriented)
std::vector<Vector3> generate_normals(const std::vector<Vector3>& points, const Neighbors_t& neigh);

// Project a neighborhood to 2D tangent plane
// The output is in correspondence with `neigh`, with the center point implicitly at (0,0)
std::vector<std::vector<Vector2>> generate_coords_projection(const std::vector<Vector3>& points,
                                                             const std::vector<Vector3> normals,
                                                             const Neighbors_t& neigh);


struct LocalTriangulationResult {
  // all triangle indices are in to the neighbors list
  std::vector<std::vector<std::array<size_t, 3>>> pointTriangles; // the triangles which touch the center vertex, always
                                                                  // numbered such that the center vertex comes first
};

LocalTriangulationResult build_delaunay_triangulations(const std::vector<std::vector<Vector2>>& coords,
                                                       const Neighbors_t& neigh);
