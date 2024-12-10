#include "point_position_geometry_ext.h"
#include "local_triangulation_ext.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/simple_idt.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace geometrycentral{
namespace pointcloud{

PointPositionGeometryExt::PointPositionGeometryExt(PointCloud& mesh, const PointData<Vector3>& positions)
    : PointPositionGeometry(mesh, positions){ }

PointPositionGeometryExt::PointPositionGeometryExt(PointCloud& mesh)
    : PointPositionGeometry(mesh){ }

void PointPositionGeometryExt::reset_neighborhoods(std::vector<std::vector<size_t>>& neighbors_){
    neighbors.reset(new NeighborhoodsExt(cloud, positions, kNeighborSize, neighbors_));
}

void PointPositionGeometryExt::computeNormals(){
  neighborsQ.ensureHave();

  normals = PointData<Vector3>(cloud);

  for (Point p : cloud.points()) {
    size_t nNeigh = neighbors->neighbors[p].size();
    if(nNeigh == 0){Vector3 N{1, 0, 0}; normals[p] = N; continue;}
    Vector3 center = positions[p];
    Eigen::MatrixXd localMat(3, nNeigh);

    for (size_t iN = 0; iN < nNeigh; iN++) {
      Vector3 neighPos = positions[neighbors->neighbors[p][iN]] - center;
      localMat(0, iN) = neighPos.x;
      localMat(1, iN) = neighPos.y;
      localMat(2, iN) = neighPos.z;
    }

    // Smallest singular vector is best normal
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(localMat, Eigen::ComputeThinU);
    Eigen::Vector3d bestNormal = svd.matrixU().col(2);

    Vector3 N{bestNormal(0), bestNormal(1), bestNormal(2)};
    N = unit(N);
    normals[p] = N;
  }
}

void PointPositionGeometryExt::computeTuftedTriangulation() {
  neighborsQ.ensureHave();
  tangentCoordinatesQ.ensureHave();

  using namespace surface;

  PointData<std::vector<std::array<Point, 3>>> localTriPoint = buildLocalTriangulations(cloud, *this, true);

  // == Make a mesh
  std::vector<std::vector<size_t>> allTris = handleToFlatInds(cloud, localTriPoint);
  std::vector<Vector3> posRaw(cloud.nPoints());
  for (size_t iP = 0; iP < posRaw.size(); iP++) {
    posRaw[iP] = positions[iP];
  }

  // Make a mesh, read off its
  std::unique_ptr<VertexPositionGeometry> posGeom;
  std::tie(tuftedMesh, posGeom) = makeSurfaceMeshAndGeometry(allTris, posRaw);
  posGeom->requireEdgeLengths();
  EdgeData<double> tuftedEdgeLengths = posGeom->edgeLengths;

  // Mollify
  mollifyIntrinsic(*tuftedMesh, tuftedEdgeLengths, 1e-5);

  // Build the cover
  buildIntrinsicTuftedCover(*tuftedMesh, tuftedEdgeLengths);

  flipToDelaunay(*tuftedMesh, tuftedEdgeLengths);

  // Create the geometry object
  tuftedGeom.reset(new EdgeLengthGeometry(*tuftedMesh, tuftedEdgeLengths));
}
}

}