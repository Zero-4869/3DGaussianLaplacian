#include "mesh_utilities.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
template <typename T>
int sign(const T& x){
    if(x > 0){return 1;}
    else{return -1;}
}

double dot2(Vector3 v){
    return dot(v, v);
}

std::tuple<Vector3, double> ProjectionOntoTriangle(Vector3 point, std::vector<Vector3> vertices, Vector3 norm_face){
    double distance_to_surface_signed = dot(point-vertices[0], norm_face);
    double distance_to_surface = std::abs(distance_to_surface_signed);
    Vector3 projected_point_surface = point - distance_to_surface_signed * norm_face + vertices[0];
    Vector3 v10 = vertices[1] - vertices[0]; Vector3 v21 = vertices[2] - vertices[1]; Vector3 v02 = vertices[0] - vertices[2];
    Vector3 p0 = projected_point_surface - vertices[0];
    Vector3 p1 = projected_point_surface - vertices[1];
    Vector3 p2 = projected_point_surface - vertices[2];
    if(sign(dot(cross(p0, v10), cross(p0, -v02))) + sign(dot(cross(p1, v21), cross(p1, -v10))) == -2){
        return std::make_tuple(projected_point_surface, distance_to_surface);
    }// inside
    else{
        double d0 = dot2(p0 - v10 * std::clamp(dot(p0, v10)/dot2(v10), 0.0, 1.0));
        double d1 = dot2(p1 - v21 * std::clamp(dot(p1, v21)/dot2(v21), 0.0, 1.0));
        double d2 = dot2(p2 - v02 * std::clamp(dot(p2, v02)/dot2(v02), 0.0, 1.0));
        if(d0 < d1){
            if(d0 < d2){
                projected_point_surface = v10 * std::clamp(dot(p0, v10)/dot2(v10), 0.0, 1.0) + vertices[0];
            }
            else{
                projected_point_surface = v02 * std::clamp(dot(p2, v02)/dot2(v02), 0.0, 1.0) + vertices[2];
            }
        }
        else{
            if(d1 < d2){
                projected_point_surface = v21 * std::clamp(dot(p1, v21)/dot2(v21), 0.0, 1.0) + vertices[1];
            }
            else{
                projected_point_surface = v02 * std::clamp(dot(p2, v02)/dot2(v02), 0.0, 1.0) + vertices[2];
            }
        }
    }
    distance_to_surface = std::abs(dot(point - projected_point_surface, norm_face));
    return std::make_tuple(projected_point_surface, distance_to_surface);

}

std::tuple<std::vector<Vector3>, std::vector<double>> ProjectionOntoMesh(const std::vector<Vector3>& points, SurfaceMesh& mesh, EmbeddedGeometryInterface& geometry){
    std::vector<Vector3> projectedPoints;
    std::vector<double> distances;
    geometry.requireFaceNormals();
    geometry.requireVertexPositions();
    for(size_t iPt = 0;iPt < points.size(); iPt++){
        Vector3 point = points[iPt];
        Vector3 point_proj, point_tmp;
        double min_dist = 1e4;
        double dist_tmp;
        for(Face f : mesh.faces()){
            Vector3 norm = geometry.faceNormals[f];
            std::vector<Vector3> vertices_triangle;
            for(Vertex v: f.adjacentVertices()){
                vertices_triangle.emplace_back(geometry.vertexPositions[v]);
            }
            const auto proj_f = ProjectionOntoTriangle(point, vertices_triangle, norm);
            point_tmp = std::get<0>(proj_f);
            dist_tmp = std::get<1>(proj_f);
            if(dist_tmp < min_dist){
                min_dist = dist_tmp;
                point_proj = point_tmp;
            }
        }
        projectedPoints.emplace_back(point_proj);
        distances.emplace_back(min_dist);
    }
    return std::make_tuple(projectedPoints, distances);
}
