#pragma once

#include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/embedded_geometry_interface.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using geometrycentral::Vector3;

std::tuple<std::vector<Vector3>, std::vector<double>> ProjectionOntoMesh(const std::vector<Vector3>& points, SurfaceMesh& mesh, EmbeddedGeometryInterface& geometry);