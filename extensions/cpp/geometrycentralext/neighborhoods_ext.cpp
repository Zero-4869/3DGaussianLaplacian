#include "neighborhoods_ext.h"
#include "geometrycentral/pointcloud/point_cloud.h"

#include "geometrycentral/utilities/knn.h"
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
namespace geometrycentral{
namespace pointcloud{

NeighborhoodsExt::NeighborhoodsExt(PointCloud& cloud_, const PointData<Vector3>& positions, unsigned int nNeighbors)
: Neighborhoods(cloud_, positions, nNeighbors){ }


NeighborhoodsExt::NeighborhoodsExt(PointCloud& cloud_, const PointData<Vector3>& positions, unsigned int nNeighbors, 
                                  std::vector<std::vector<size_t>>& neighbors_)
: Neighborhoods(cloud_, positions, nNeighbors)
{
    for(Point p : cloud.points()){
        size_t pInd = p.getIndex();
        size_t nNeigh = neighbors_[pInd].size();
        if(nNeigh == 0){neighbors[p].resize(nNeigh); continue;}
        std::vector<size_t> neighInd = neighbors_[pInd];
        neighbors[p].resize(neighInd.size());
        for(size_t j = 0; j < neighInd.size(); j++) {
            neighbors[p][j] = cloud.point(neighInd[j]);
        }
    }
}

}
}
