#pragma once

#include "neighborhoods_ext.h"
// #include "geometrycentral/pointcloud/point_cloud.h"
// #include "geometrycentral/surface/edge_length_geometry.h"
// #include "geometrycentral/utilities/dependent_quantity.h"
// #include "geometrycentral/utilities/vector2.h"
// #include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"

namespace geometrycentral{

namespace pointcloud{

class PointPositionGeometryExt: public PointPositionGeometry{
public:
    PointPositionGeometryExt(PointCloud& mesh, const PointData<Vector3>& positions);
    PointPositionGeometryExt(PointCloud& mesh);
    void reset_neighborhoods(std::vector<std::vector<size_t>> &neighbors_);

protected:
    virtual void computeTuftedTriangulation();
    virtual void computeNormals();
    
};

}
}

