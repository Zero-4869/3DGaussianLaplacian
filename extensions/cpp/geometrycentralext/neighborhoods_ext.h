#pragma once
#include "geometrycentral/pointcloud/neighborhoods.h"

namespace geometrycentral{
namespace pointcloud{

class NeighborhoodsExt:public Neighborhoods{
public:
    NeighborhoodsExt(PointCloud& cloud_, const PointData<Vector3>& positions, unsigned int nNeighbors);
    NeighborhoodsExt(PointCloud& cloud, const PointData<Vector3>& positions, unsigned int nNeighbors,
                        std::vector<std::vector<size_t>>& neighbors_);
};
}
}