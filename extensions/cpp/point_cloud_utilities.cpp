#include "point_cloud_utilities.h"

#include "geometrycentral/utilities/elementary_geometry.h"
#include "geometrycentral/utilities/knn.h"

#include "Eigen/Dense"

#include <cfloat>
#include <numeric>
#include <algorithm>
#include <vector>

std::vector<std::vector<size_t>> generate_knn(const std::vector<Vector3>& points, size_t k) {

  geometrycentral::NearestNeighborFinder finder(points);

  std::vector<std::vector<size_t>> result;
  for (size_t i = 0; i < points.size(); i++) {
    result.emplace_back(finder.kNearestNeighbors(i, k));
  }

  return result;
}

/**
 * Argsort(currently support ascending sort)
 * @tparam T array element type
 * @param array input array
 * @return indices w.r.t sorted array
 */
template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

std::vector<std::vector<size_t>> generate_knn_adapted(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse) {

  std::vector<std::vector<size_t>> result;
  for(size_t i = 0; i < points.size(); i++) {
    Eigen::MatrixXd rt_inv(3, 3);
    rt_inv(0, 0) = RT_inverse.coeff(i, 0);rt_inv(0, 1) = RT_inverse.coeff(i, 1);rt_inv(0, 2) = RT_inverse.coeff(i, 2);
    rt_inv(1, 0) = RT_inverse.coeff(i, 3);rt_inv(1, 1) = RT_inverse.coeff(i, 4);rt_inv(1, 2) = RT_inverse.coeff(i, 5);
    rt_inv(2, 0) = RT_inverse.coeff(i, 6);rt_inv(2, 1) = RT_inverse.coeff(i, 7);rt_inv(2, 2) = RT_inverse.coeff(i, 8);
    Eigen::MatrixXd p(points.size(), 3);
    for(int iN = 0; iN < points.size(); iN++){
      p(iN, 0) = points[iN].x - points[i].x; p(iN, 1) = points[iN].y - points[i].y; p(iN, 2) = points[iN].z - points[i].z;
    }
    Eigen::MatrixXd weights = p * rt_inv.transpose();
    std::vector<double> dists;
    for(size_t j = 0; j < p.rows(); j++){
      dists.emplace_back(weights(j, 0) * weights(j, 0) + weights(j, 1) * weights(j, 1) + weights(j, 2) * weights(j, 2));
    }
    std::vector<size_t> indices = argsort(dists);
    std::vector<size_t> selected_indices;
    for (size_t j = 1; j < k+1; j++){
      selected_indices.emplace_back(indices[j]);
    }
    result.emplace_back(selected_indices);
  }
  return result;
}

std::vector<std::vector<size_t>> generate_knn_adapted(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse, double radius) {

  std::vector<std::vector<size_t>> result;
  for(size_t i = 0; i < points.size(); i++) {
    Eigen::MatrixXd rt_inv(3, 3);
    rt_inv(0, 0) = RT_inverse.coeff(i, 0);rt_inv(0, 1) = RT_inverse.coeff(i, 1);rt_inv(0, 2) = RT_inverse.coeff(i, 2);
    rt_inv(1, 0) = RT_inverse.coeff(i, 3);rt_inv(1, 1) = RT_inverse.coeff(i, 4);rt_inv(1, 2) = RT_inverse.coeff(i, 5);
    rt_inv(2, 0) = RT_inverse.coeff(i, 6);rt_inv(2, 1) = RT_inverse.coeff(i, 7);rt_inv(2, 2) = RT_inverse.coeff(i, 8);
    Eigen::MatrixXd p(points.size(), 3);
    for(int iN = 0; iN < points.size(); iN++){
      p(iN, 0) = points[iN].x - points[i].x; p(iN, 1) = points[iN].y - points[i].y; p(iN, 2) = points[iN].z - points[i].z;
    }
    Eigen::MatrixXd weights = p * rt_inv.transpose();
    std::vector<double> dists;
    for(size_t j = 0; j < p.rows(); j++){
      dists.emplace_back(weights(j, 0) * weights(j, 0) + weights(j, 1) * weights(j, 1) + weights(j, 2) * weights(j, 2));
    }
    std::vector<size_t> indices = argsort(dists);
    std::vector<size_t> selected_indices;
    for (size_t j = 1; j < k+1; j++){
      if(dists[indices[j]] < radius){
        selected_indices.emplace_back(indices[j]);
      }
    }
    result.emplace_back(selected_indices);
  }
  return result;
}

std::vector<std::vector<size_t>> generate_knn_mahalanobis_euclidean(const std::vector<Vector3>& points, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse) {

  std::vector<std::vector<size_t>> result;
  for(size_t i = 0; i < points.size(); i++) {
    Eigen::MatrixXd rt_inv(3, 3);
    rt_inv(0, 0) = RT_inverse.coeff(i, 0);rt_inv(0, 1) = RT_inverse.coeff(i, 1);rt_inv(0, 2) = RT_inverse.coeff(i, 2);
    rt_inv(1, 0) = RT_inverse.coeff(i, 3);rt_inv(1, 1) = RT_inverse.coeff(i, 4);rt_inv(1, 2) = RT_inverse.coeff(i, 5);
    rt_inv(2, 0) = RT_inverse.coeff(i, 6);rt_inv(2, 1) = RT_inverse.coeff(i, 7);rt_inv(2, 2) = RT_inverse.coeff(i, 8);
    Eigen::MatrixXd p(points.size(), 3);
    for(int iN = 0; iN < points.size(); iN++){
      p(iN, 0) = points[iN].x - points[i].x; p(iN, 1) = points[iN].y - points[i].y; p(iN, 2) = points[iN].z - points[i].z;
    }
    Eigen::MatrixXd weights = p * rt_inv.transpose();
    std::vector<double> dists;
    for(size_t j = 0; j < p.rows(); j++){
      double dist_euclidean = p(j, 0) * p(j, 0) + p(j, 1) * p(j, 1) + p(j, 2) * p(j, 2);
      double dist_mahalanobis = weights(j, 0) * weights(j, 0) + weights(j, 1) * weights(j, 1) + weights(j, 2) * weights(j, 2); 
      dists.emplace_back(dist_euclidean + dist_mahalanobis);
    }
    std::vector<size_t> indices = argsort(dists);
    std::vector<size_t> selected_indices;
    for (size_t j = 1; j < k+1; j++){
      selected_indices.emplace_back(indices[j]);
    }
    result.emplace_back(selected_indices);
  }
  return result;
}


std::vector<std::vector<size_t>> filter_mahalanobis(const std::vector<Vector3>& points, const std::vector<std::vector<size_t>>& pointInd, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse) {

  std::vector<std::vector<size_t>> result;
  for(size_t i = 0; i < points.size(); i++) {
    Eigen::MatrixXd rt_inv(3, 3);
    rt_inv(0, 0) = RT_inverse.coeff(i, 0);rt_inv(0, 1) = RT_inverse.coeff(i, 1);rt_inv(0, 2) = RT_inverse.coeff(i, 2);
    rt_inv(1, 0) = RT_inverse.coeff(i, 3);rt_inv(1, 1) = RT_inverse.coeff(i, 4);rt_inv(1, 2) = RT_inverse.coeff(i, 5);
    rt_inv(2, 0) = RT_inverse.coeff(i, 6);rt_inv(2, 1) = RT_inverse.coeff(i, 7);rt_inv(2, 2) = RT_inverse.coeff(i, 8);
    Eigen::MatrixXd p(pointInd[i].size(), 3);
    for(int iN = 0; iN < pointInd[i].size(); iN++){
      p(iN, 0) = points[pointInd[i][iN]].x - points[i].x; p(iN, 1) = points[pointInd[i][iN]].y - points[i].y; p(iN, 2) = points[pointInd[i][iN]].z - points[i].z;
    }
    Eigen::MatrixXd weights = p * rt_inv.transpose();
    std::vector<double> dists;
    for(size_t j = 0; j < p.rows(); j++){
      double dist_mahalanobis = weights(j, 0) * weights(j, 0) + weights(j, 1) * weights(j, 1) + weights(j, 2) * weights(j, 2); 
      dists.emplace_back(dist_mahalanobis);
    }
    std::vector<size_t> indices = argsort(dists);
    std::vector<size_t> selected_indices;
    for (size_t j = 0; j < k; j++){
      selected_indices.emplace_back(pointInd[i][indices[j]]);
    }
    result.emplace_back(selected_indices);
  }
  return result;
}

std::vector<std::vector<size_t>> filter_mahalanobis_bilateral(const std::vector<Vector3>& points, const std::vector<std::vector<size_t>>& pointInd, size_t k, const geometrycentral::DenseMatrix<double> & RT_inverse) {

  std::vector<std::vector<size_t>> result;
  std::vector<std::vector<size_t>> pointInd_ordered;
  for(size_t i = 0; i < points.size(); i++) {
    Eigen::MatrixXd rt_inv(3, 3);
    rt_inv(0, 0) = RT_inverse.coeff(i, 0);rt_inv(0, 1) = RT_inverse.coeff(i, 1);rt_inv(0, 2) = RT_inverse.coeff(i, 2);
    rt_inv(1, 0) = RT_inverse.coeff(i, 3);rt_inv(1, 1) = RT_inverse.coeff(i, 4);rt_inv(1, 2) = RT_inverse.coeff(i, 5);
    rt_inv(2, 0) = RT_inverse.coeff(i, 6);rt_inv(2, 1) = RT_inverse.coeff(i, 7);rt_inv(2, 2) = RT_inverse.coeff(i, 8);
    Eigen::MatrixXd p(pointInd[i].size(), 3);
    for(int iN = 0; iN < pointInd[i].size(); iN++){
      p(iN, 0) = points[pointInd[i][iN]].x - points[i].x; p(iN, 1) = points[pointInd[i][iN]].y - points[i].y; p(iN, 2) = points[pointInd[i][iN]].z - points[i].z;
    }
    Eigen::MatrixXd weights = p * rt_inv.transpose();
    std::vector<double> dists;
    for(size_t j = 0; j < p.rows(); j++){
      double dist_mahalanobis = weights(j, 0) * weights(j, 0) + weights(j, 1) * weights(j, 1) + weights(j, 2) * weights(j, 2); 
      dists.emplace_back(dist_mahalanobis);
    }
    std::vector<size_t> indices = argsort(dists);
    std::vector<size_t> selected_indices;
    for(size_t indice:indices){selected_indices.emplace_back(pointInd[i][indice]);}
    pointInd_ordered.emplace_back(selected_indices);
  }

  for(size_t i = 0; i < points.size(); i++) { 
    std::vector<size_t> selected_indices;
    for (size_t j = 0; j < k; j++){
      bool bilateral = false;
      size_t current_ind = pointInd_ordered[i][j];
      for(size_t t = 0; t < k; t++){
        if(pointInd_ordered[current_ind][t] == i){bilateral = true; break;}
      }
      if(bilateral){selected_indices.emplace_back(current_ind);}
    }
    result.emplace_back(selected_indices);
  }
  return result;
}


// std::vector<Vector3> filter_points(const std::vector<Vector3>& points, const std::vector<double>& opacity){
//   std::vector<Vector3> new_points;
//   for(size_t iPt = 0; iPt < points.size();iPt++){
//     if(opacity[iPt] > 0.5){
//       new_points.emplace_back(points[iPt]);
//     }
//   }
//   return new_points;
// }

std::vector<std::vector<size_t>> filter_knn(const std::vector<Vector3>& points, Neighbors_t& neigh, const std::vector<Vector3> normals, const size_t N){
  std::vector<std::vector<size_t>> result;
  double inner_product;
  bool print = true;
  for(size_t iPt = 0;iPt < points.size();iPt++){
    std::vector<size_t> tt;
    std::vector<double> disparity;
    Vector3 center = points[iPt];
    Vector3 normal = normals[iPt];
    size_t nNeigh = neigh[iPt].size();
    // double threshold = thresholds[iPt];
    for(size_t iN = 0; iN < nNeigh; iN++){
      Vector3 x = points[neigh[iPt][iN]] - center;
      inner_product = x[0] * normal[0] + x[1] * normal[1] + x[2] * normal[2];
      disparity.emplace_back(std::abs(inner_product));
      // if(std::abs(inner_product) <= threshold){
      //   tt.emplace_back(iN);
      // }
    }
    std::vector<size_t> index_sorted = argsort(disparity);
    for(size_t iN = 0; iN < N; iN++){
      tt.emplace_back(neigh[iPt][index_sorted[iN]]);
    }
    result.emplace_back(tt);
  }
  return result;
}

std::vector<Vector3> generate_normals(const std::vector<Vector3>& points, const Neighbors_t& neigh) {

  std::vector<Vector3> normals(points.size());

  for (size_t iPt = 0; iPt < points.size(); iPt++) {
    size_t nNeigh = neigh[iPt].size();
    if(nNeigh == 0){Vector3 N{1., 0., 0.}; normals[iPt] = N; continue;}
    // Compute centroid
    Vector3 center{0., 0., 0.};
    // for (size_t iN = 0; iN < nNeigh; iN++) {
    //   center += points[neigh[iPt][iN]];
    // }
    // center /= nNeigh + 1;
    center = points[iPt];

    // Assemble matrix os vectors from centroid
    Eigen::MatrixXd localMat(3, neigh[iPt].size());
    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector3 neighPos = points[neigh[iPt][iN]] - center;
      localMat(0, iN) = neighPos.x;
      localMat(1, iN) = neighPos.y;
      localMat(2, iN) = neighPos.z;
    }

    // Smallest singular vector is best normal
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(localMat, Eigen::ComputeThinU);
    Eigen::Vector3d bestNormal = svd.matrixU().col(2);

    Vector3 N{bestNormal(0), bestNormal(1), bestNormal(2)};
    N = unit(N);
    normals[iPt] = N;
  }

  return normals;
}


std::vector<std::vector<Vector2>> generate_coords_projection(const std::vector<Vector3>& points,
                                                             const std::vector<Vector3> normals,
                                                             const Neighbors_t& neigh) {
  std::vector<std::vector<Vector2>> coords(points.size());

  for (size_t iPt = 0; iPt < points.size(); iPt++) {
    size_t nNeigh = neigh[iPt].size();
    coords[iPt].resize(nNeigh);
    Vector3 center = points[iPt];
    Vector3 normal = normals[iPt];

    // build an arbitrary tangent basis
    Vector3 basisX, basisY;
    auto r = normal.buildTangentBasis();
    basisX = r[0];
    basisY = r[1];

    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector3 vec = points[neigh[iPt][iN]] - center;
      vec = vec.removeComponent(normal);

      Vector2 coord{dot(basisX, vec), dot(basisY, vec)};
      coords[iPt][iN] = coord;
    }
  }

  return coords;
}

// For each planar-projected neighborhood, generate the triangles in the Delaunay triangulation which are incident on
// the center vertex.
//
// This could be done robustly via e.g. Shewchuk's triangle.c. However, instead we use a simple self-contained strategy
// which leverages the needs of this particular situation. In particular, we don't really care about getting exactly the
// Delaunay triangulation; we're just looking for any sane triangulation to use as input the the subsequent step. We
// just use Delaunay because we like the property that (in the limit of sampling), it's a triple-cover of the domain;
// with other strategies it's hard to quantify how many times our triangles cover the domain. This makes the problem
// easier, because for degenerate/underdetermined cases, we're happy to output any triangulation, even if it's not the
// Delaunay triangulation in exact arithmetic.
//
// This strategy works by angularly sorting points relative to the neighborhood center, then walking around circle
// identifying pairs of edges which form Delaunay triangles (more details inline). In particular, using a sorting of the
// points helps to distinguish indeterminate cases and always output some triangles. Additionally, a few heuristics are
// included for handling of degenerate and collinear points. This routine has O(n*k^2) complexity, where k is the
// neighborhood size).
LocalTriangulationResult build_delaunay_triangulations(const std::vector<std::vector<Vector2>>& coords,
                                                       const Neighbors_t& neigh) {

  // A few innocent numerical parameters
  const double PERTURB_THRESH = 1e-7;         // in units of relative length
  const double ANGLE_COLLINEAR_THRESH = 1e-5; // in units of radians
  const double OUTSIDE_EPS = 1e-4;            // in units of relative length

  // NOTE: This is not robust if the entire neighbohood is coincident (or very nearly coincident) with the centerpoint.
  // Though in that case, the generate_normals() routine will probably also have issues.

  size_t nPts = coords.size();
  LocalTriangulationResult result;
  result.pointTriangles.resize(nPts);

  for (size_t iPt = 0; iPt < nPts; iPt++) {
    size_t nNeigh = neigh[iPt].size();
    if(nNeigh == 0){continue;}
    double lenScale = norm(coords[iPt].back());

    // Something is hopelessly degenerate, don't even bother trying. No triangles for this point.
    if (!std::isfinite(lenScale) || lenScale <= 0) {
      continue;
    }

    // Local copies of points
    std::vector<Vector2> perturbPoints = coords[iPt];
    std::vector<size_t> perturbInds = neigh[iPt];

    { // Perturb points which are extremely close to the source
      for (size_t iNeigh = 0; iNeigh < nNeigh; iNeigh++) {
        Vector2& neighPt = perturbPoints[iNeigh];
        double dist = norm(neighPt);
        if (dist < lenScale * PERTURB_THRESH) { // need to perturb
          Vector2 dir = normalize(neighPt);
          if (!isfinite(dir)) { // even direction is degenerate :(
            // pick a direction from index
            double thetaDir = (2. * M_PI * iNeigh) / nNeigh;
            dir = Vector2::fromAngle(thetaDir);
          }

          // Set the distance from the origin for the pertubed point. Including the index avoids creating many
          // co-circular points; no need to stress the Delaunay triangulation unnessecarily.
          double len = (1. + static_cast<double>(iNeigh) / nNeigh) * lenScale * PERTURB_THRESH * 10;

          neighPt = len * dir; // update the point
        }
      }
    }


    size_t closestPointInd = 0;
    double closestPointDist = std::numeric_limits<double>::infinity();
    bool hasBoundary = false;
    { // Find the starting point for the angular search.
      // If there is boundary, it's the beginning of the interior region; otherwise its the closest point.
      // (either way, this point is guaranteed to appear in the triangulation)
      // NOTE: boundary check is actually done after inline sort below, since its cheaper there

      for (size_t iNeigh = 0; iNeigh < nNeigh; iNeigh++) {
        Vector2 neighPt = perturbPoints[iNeigh];
        double thisPointDist = norm(neighPt);
        if (thisPointDist < closestPointDist) {
          closestPointDist = thisPointDist;
          closestPointInd = iNeigh;
        }
      }
    }


    std::vector<size_t> sortInds(nNeigh);
    { // = Angularly sort the points CCW, such that the closest point comes first

      // Angular sort
      std::vector<double> pointAngles(nNeigh);
      for (size_t i = 0; i < nNeigh; i++) {
        pointAngles[i] = arg(perturbPoints[i]);
      }
      std::iota(std::begin(sortInds), std::end(sortInds), 0);
      std::sort(sortInds.begin(), sortInds.end(),
                [&](const size_t& a, const size_t& b) -> bool { return pointAngles[a] < pointAngles[b]; });

      // Check if theres a gap of >= PI between any two consecutive points. If so it's a boundary.
      double largestGap = -1;
      size_t largestGapEndInd = 0;
      for (size_t i = 0; i < nNeigh; i++) {
        size_t j = (i + 1) % nNeigh;
        double angleI = pointAngles[sortInds[i]];
        double angleJ = pointAngles[sortInds[j]];
        double gap;
        if (i + 1 == nNeigh) {
          gap = angleJ - (angleI + 2 * M_PI);
        } else {
          gap = angleJ - angleI;
        }

        if (gap > largestGap) {
          largestGap = gap;
          largestGapEndInd = j;
        }
      }

      // The start of the cyclic ordering is either
      size_t firstInd;
      if (largestGap > (M_PI - ANGLE_COLLINEAR_THRESH)) {
        firstInd = largestGapEndInd;
        hasBoundary = true;
      } else {
        firstInd = std::distance(sortInds.begin(), std::find(sortInds.begin(), sortInds.end(), closestPointInd));
        hasBoundary = false;
      }

      // Cyclically permute to ensure starting point comes first
      std::rotate(sortInds.begin(), sortInds.begin() + firstInd, sortInds.end());
    }

    size_t edgeStartInd = 0;
    std::vector<std::array<size_t, 3>>& thisPointTriangles = result.pointTriangles[iPt]; // accumulate result

    // end point should wrap around the check the first point only if there is no boundary
    size_t searchEnd = nNeigh + (hasBoundary ? 0 : 1);

    // Walk around the angularly-sorted points, forming triangles spanning angular regions. To construct each triangle,
    // we start with leg at edgeStartInd, then search over edgeEndInd to find the first other end which has an empty
    // circumcircle. Once it is found, we form a triangle and being searching again from edgeEndInd.
    //
    // At first, this might sound like it has n^3 complexity, since there are n^2 triangles to consider, and testing
    // each costs n. However, since we march around the angular direction in increasing order, we will only test at most
    // O(n) triangles, leading to n^2 complexity.
    while (edgeStartInd < nNeigh) {
      size_t iStart = sortInds[edgeStartInd];
      Vector2 startPos = perturbPoints[iStart];

      // lookahead and find the first triangle we can form with an empty (or nearly empty) circumcircle
      bool foundTri = false;
      for (size_t edgeEndInd = edgeStartInd + 1; edgeEndInd < searchEnd; edgeEndInd++) {
        size_t iEnd = sortInds[edgeEndInd % nNeigh];
        Vector2 endPos = perturbPoints[iEnd];

        // If the start and end points are too close to being colinear, don't bother
        Vector2 startPosDir = unit(startPos);
        Vector2 endPosDir = unit(endPos);
        if (std::fabs(cross(startPosDir, endPosDir)) < ANGLE_COLLINEAR_THRESH) {
          continue;
        }

        // Find the circumcenter and circumradius
        geometrycentral::RayRayIntersectionResult2D isect =
            rayRayIntersection(0.5 * startPos, startPosDir.rotate90(), 0.5 * endPos, -endPosDir.rotate90());
        Vector2 circumcenter = 0.5 * startPos + isect.tRay1 * startPosDir.rotate90();
        double circumradius = norm(circumcenter);

        // Find the minimum distance to the circumcenter
        double nearestDistSq = std::numeric_limits<double>::infinity();
        double circumradSqConservative = (circumradius - lenScale * OUTSIDE_EPS);
        circumradSqConservative *= circumradSqConservative;
        for (size_t iTest = 0; iTest < nNeigh; iTest++) {
          if (iTest == iStart || iTest == iEnd) continue; // skip the points forming the triangle
          double thisDistSq = norm2(circumcenter - perturbPoints[iTest]);
          nearestDistSq = std::fmin(nearestDistSq, thisDistSq);

          // if it's already strictly inside, no need to keep searching
          if (nearestDistSq < circumradSqConservative) break;
        }
        double nearestDist = std::sqrt(nearestDistSq);

        // Accept the triangle if its circumcircle is sufficiently empty
        // NOTE: The choice of signs in this expression is important: we preferential DO accept triangles whos
        // circumcircle is barely empty. This makes sense here because our circular loop already avoids any risk of
        // accepting overlapping triangles; the risk is in not accepting any, so we should preferrentially accept.
        if (nearestDist + lenScale * OUTSIDE_EPS > circumradius) {
          std::array<size_t, 3> triInds = {std::numeric_limits<size_t>::max(), iStart, iEnd};
          thisPointTriangles.push_back(triInds);

          // advance the circular search to find a triangle starting at this edge
          edgeStartInd = edgeEndInd;
          foundTri = true;
          break;
        }
      }

      // if we couldn't find any triangles, increment the start index
      if (!foundTri) {
        edgeStartInd++;
      }
    }
  }

  return result;
}