#pragma once

#include "geometrycentral/pointcloud/point_cloud.h"
#include "point_position_geometry_ext.h"

namespace geometrycentral {
namespace pointcloud {

// === Readers ===

// Read from a file by name. Type can be optionally inferred from filename.
std::tuple<std::unique_ptr<PointCloud>, std::unique_ptr<PointPositionGeometryExt>> readPointCloudExt(std::string filename,
                                                                                               std::string type = "");

// Same as above, but from an istream. Must specify type.
std::tuple<std::unique_ptr<PointCloud>, std::unique_ptr<PointPositionGeometryExt>> readPointCloudExt(std::istream& in,
                                                                                               std::string type);

std::tuple<std::unique_ptr<PointCloud>, std::unique_ptr<PointPositionGeometryExt>> readPointCloudExt(const std::vector<std::array<double, 3>>& vPos);
} // namespace pointcloud
} // namespace geometrycentral