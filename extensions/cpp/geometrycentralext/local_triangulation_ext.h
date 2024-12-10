#pragma once

#include "geometrycentral/pointcloud/local_triangulation.h"

namespace geometrycentral {
namespace pointcloud {

PointData<std::vector<std::array<Point, 3>>> buildLocalTriangulationsExt(PointCloud& cloud, PointPositionGeometry& geom,
                                                                      bool withDegeneracyHeuristic = true);

// Convert a local neighbor indexed list to global indices

// PointData<std::vector<std::array<size_t, 3>>>
// handleToInds(PointCloud& cloud, const PointData<std::vector<std::array<Point, 3>>>& handleResult);

// std::vector<std::vector<size_t>> handleToFlatInds(PointCloud& cloud,
//                                                   const PointData<std::vector<std::array<Point, 3>>>& handleResult);


} // namespace pointcloud
} // namespace geometrycentral