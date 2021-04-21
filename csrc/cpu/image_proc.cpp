#include "cpu/image_proc.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>

namespace helper {

    bool in_bounds(Eigen::Vector2f p, int h, int w) {
        return p.x() >= 0.0 && p.x() < w && p.y() >= 0.0 && p.y() < h;
    }

    bool valid_flow_at(const Eigen::Vector2f& p, int h, int w, py::array_t<float>& flow_image, Eigen::Vector2f& flow) {
        if (!in_bounds(p, h, w)) {
            return false;
        }

        flow.x() = *flow_image.data(p.y(), p.x(), 0);
        flow.y() = *flow_image.data(p.y(), p.x(), 1);

        if (!flow.allFinite()) {
            return false;
        }

        return true;
    }
}

namespace image_proc {

    using Vec2f = Eigen::Vector2f;
    
    template <class V, class L = std::less<std::string>, 
              class A = Eigen::aligned_allocator<std::pair<const std::string, V>>>
    using aligned_dict = std::map<std::string, V, L, A>;

    py::array_t<float> compute_augmented_flow_from_rotation(py::array_t<float>& flow_image_rot_sa2so, 
                                              py::array_t<float>& flow_image_so2to, 
                                              py::array_t<float>& flow_image_rot_to2ta,
                                              const int h, const int w) {
        // TODO: change to runtime asserts
        // assert(flow_image_rot_sa2so.ndim() == 3);
        // assert(flow_image_rot_sa2so.shape(0) == 2);
        // assert(flow_image_rot_sa2so.shape(1) == h);
        // assert(flow_image_rot_sa2so.shape(2) == w);

        // assert(flow_image_so2to.ndim() == 3);
        // assert(flow_image_so2to.shape(0) == 2);
        // assert(flow_image_so2to.shape(1) == h);
        // assert(flow_image_so2to.shape(2) == w);

        // assert(flow_image_rot_to2ta.ndim() == 3);
        // assert(flow_image_rot_to2ta.shape(0) == 2);
        // assert(flow_image_rot_to2ta.shape(1) == h);
        // assert(flow_image_rot_to2ta.shape(2) == w);

        // allocate memory for output array
        py::array_t<float> flow_image_rot_sa2ta = py::array_t<float>(flow_image_rot_sa2so.request().size);

        // reshape array to match input shape
        flow_image_rot_sa2ta.resize({h, w, 2});

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                // update output flow image
                *flow_image_rot_sa2ta.mutable_data(y, x, 0) = - std::numeric_limits<float>::infinity();
                *flow_image_rot_sa2ta.mutable_data(y, x, 1) = - std::numeric_limits<float>::infinity();

                Vec2f p_sa(x, y);

                /////////////////////////////////////////////////////////////////////////////////
                // 1. SOURCE AUGMENTED TO SOURCE ORIGINAL
                /////////////////////////////////////////////////////////////////////////////////

                // flow from source augmented to source original
                Vec2f flow_sa2so(*flow_image_rot_sa2so.data(y, x, 0), *flow_image_rot_sa2so.data(y, x, 1));
                
                // flow_sa2so should be dense and w/o any invalid value
                if (!flow_sa2so.allFinite()) {
                    throw "flow_sa2so sould be dense and w/o any invalid values!";
                }

                // compute warped location on source original (so we're going from source augmented to source original)
                Vec2f p_so = p_sa + flow_sa2so;

                // init flow_sa2ta with the first contribution, i.e, flow_sa2so
                Vec2f flow_sa2ta = flow_sa2so;

                /////////////////////////////////////////////////////////////////////////////////
                // 2. SOURCE ORIGINAL TO TARGET ORIGINAL
                /////////////////////////////////////////////////////////////////////////////////
                int u0 = std::floor(p_so.x());
                int u1 = u0 + 1;
                int v0 = std::floor(p_so.y());
                int v1 = v0 + 1;

                Vec2f p00(u0, v0);
                Vec2f p01(u0, v1);
                Vec2f p10(u1, v0);
                Vec2f p11(u1, v1);

                aligned_dict<Vec2f> valid_coords;
                aligned_dict<Vec2f> valid_flows;

                Vec2f flow_00_so2to;
                if (helper::valid_flow_at(p00, h, w, flow_image_so2to, flow_00_so2to)) {
                    valid_coords["p00"] = p00;
                    valid_flows["p00"]  = flow_00_so2to;
                }

                Vec2f flow_01_so2to;
                if (helper::valid_flow_at(p01, h, w, flow_image_so2to, flow_01_so2to)) {
                    valid_coords["p01"] = p01;
                    valid_flows["p01"]  = flow_01_so2to;
                }

                Vec2f flow_10_so2to;
                if (helper::valid_flow_at(p10, h, w, flow_image_so2to, flow_10_so2to)) {
                    valid_coords["p10"] = p10;
                    valid_flows["p10"]  = flow_10_so2to;
                }

                Vec2f flow_11_so2to;
                if (helper::valid_flow_at(p11, h, w, flow_image_so2to, flow_11_so2to)) {
                    valid_coords["p11"] = p11;
                    valid_flows["p11"]  = flow_11_so2to;
                }

                // Depending on how many valid flows we have, do bilinear interpolation or nearest neighbor:
                Vec2f flow_so2to;

                if (valid_coords.size() == 0) {
                    continue;
                } else if (valid_coords.size() == 4) {
                    // Bilinear interpolation
                    float du = p_so.x() - u0;
                    float dv = p_so.y() - v0;

                    float w00 = (1 - du) * (1 - dv);
                    float w01 = (1 - du) * dv;
                    float w10 = du * (1 - dv);
                    float w11 = du * dv;

                    flow_so2to = w00 * valid_flows["p00"] + 
                                 w01 * valid_flows["p01"] + 
                                 w10 * valid_flows["p10"] + 
                                 w11 * valid_flows["p11"];                    
                } else {
                    // Nearest Neighbor
                    std::string nn = "None";
                    float min_dist = std::numeric_limits<float>::max();

                    for (const auto& valid_coord : valid_coords) {
                        const std::string k = valid_coord.first;
                        const Vec2f& p = valid_coord.second;

                        float dist = (p_so - p).norm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            nn = k;
                        }
                    }
                    
                    if (nn == "None") {
                        throw std::runtime_error("Neighrest Neighbor 'nn' was not assigned...");
                    }

                    flow_so2to = valid_flows[nn];
                }

                // compute warped location on target original (so we're going from source original to target original)
                Vec2f p_to = p_so + flow_so2to;

                // add flow_so2to to flow_sa2ta
                flow_sa2ta += flow_so2to;

                /////////////////////////////////////////////////////////////////////////////////
                // 3. TARGET ORIGINAL TO TARGET AUGMENTED
                /////////////////////////////////////////////////////////////////////////////////
                u0 = std::floor(p_to.x());
                u1 = u0 + 1;
                v0 = std::floor(p_to.y());
                v1 = v0 + 1;

                p00 = Vec2f(u0, v0);
                p01 = Vec2f(u0, v1);
                p10 = Vec2f(u1, v0);
                p11 = Vec2f(u1, v1);

                valid_coords.clear();
                valid_flows.clear();

                Vec2f flow_00_to2ta;
                if (helper::valid_flow_at(p00, h, w, flow_image_rot_to2ta, flow_00_to2ta)) {
                    valid_coords["p00"] = p00;
                    valid_flows["p00"]  = flow_00_to2ta;
                }

                Vec2f flow_01_to2ta;
                if (helper::valid_flow_at(p01, h, w, flow_image_rot_to2ta, flow_01_to2ta)) {
                    valid_coords["p01"] = p01;
                    valid_flows["p01"]  = flow_01_to2ta;
                }

                Vec2f flow_10_to2ta;
                if (helper::valid_flow_at(p10, h, w, flow_image_rot_to2ta, flow_10_to2ta)) {
                    valid_coords["p10"] = p10;
                    valid_flows["p10"]  = flow_10_to2ta;
                }

                Vec2f flow_11_to2ta;
                if (helper::valid_flow_at(p11, h, w, flow_image_rot_to2ta, flow_11_to2ta)) {
                    valid_coords["p11"] = p11;
                    valid_flows["p11"]  = flow_11_to2ta;
                }

                // Depending on how many valid flows we have, do bilinear interpolation or nearest neighbor:
                Vec2f flow_to2ta;

                if (valid_coords.size() == 0) {
                    continue;
                } else if (valid_coords.size() == 4) {
                    // Bilinear interpolation
                    float du = p_to.x() - u0;
                    float dv = p_to.y() - v0;

                    float w00 = (1 - du) * (1 - dv);
                    float w01 = (1 - du) * dv;
                    float w10 = du * (1 - dv);
                    float w11 = du * dv;

                    flow_to2ta = w00 * valid_flows["p00"] + 
                                 w01 * valid_flows["p01"] + 
                                 w10 * valid_flows["p10"] + 
                                 w11 * valid_flows["p11"];                    
                } else {
                    // Nearest Neighbor
                    std::string nn = "None";
                    float min_dist = std::numeric_limits<float>::max();

                    for (const auto& valid_coord : valid_coords) {
                        const std::string k = valid_coord.first;
                        const Vec2f& p = valid_coord.second;

                        float dist = (p_to - p).norm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            nn = k;
                        }
                    }
                    
                    if (nn == "None") {
                        throw std::runtime_error("Neighrest Neighbor 'nn' was not assigned...");
                    }

                    flow_to2ta = valid_flows[nn];
                }

                // add flow_to2ta to flow_sa2ta
                flow_sa2ta += flow_to2ta;

                // update output flow image
                *flow_image_rot_sa2ta.mutable_data(y, x, 0) = flow_sa2ta.x();
                *flow_image_rot_sa2ta.mutable_data(y, x, 1) = flow_sa2ta.y();
            }
        }

        return flow_image_rot_sa2ta;
    }

    int count_tp1(py::array_t<bool> &p, py::array_t<bool> &gt) {
        assert(p.ndim() == 2);
        assert(gt.ndim() == 2);
        const int n_batch = p.shape(0);
        const int dimz = p.shape(1);

        auto& ptr = p;
        int counter = 0;
        for (int i = 0; i < n_batch; i++) 
            for (int z = 0 ; z < dimz ; z++)
                if (*gt.data(i, z)) {
                    counter += CHECK1();
                }
        return counter;
    }

    int count_tp2(py::array_t<bool> &p, py::array_t<bool> &gt) {
        assert(p.ndim() == 4);
        assert(gt.ndim() == 4);
        const int n_batch = p.shape(0);
        assert(p.shape(1) == 1);
        const int height = p.shape(2);
        const int width = p.shape(3);

        auto& ptr = p;
        int counter = 0;
        for (int i = 0; i < n_batch; i++) 
            for (int y = 0; y <  height; y++)
                for (int x = 0; x < width; x++) {
                    if (*gt.data(i, 0, y, x)) {
                        counter += CHECK2();
                    }
                }
        return counter;
    }

    int count_tp3(py::array_t<bool> &p, py::array_t<bool> &gt) {
        assert(p.ndim() == 5);
        assert(gt.ndim() == 5);
        const int n_batch = p.shape(0);
        assert(p.shape(1) == 1);
        const int dimz = p.shape(2);
        const int dimy = p.shape(3);
        const int dimx = p.shape(4);

        auto& ptr = p;
        int counter = 0;
        for (int i = 0; i < n_batch; i++) 
            for (int z = 0 ; z < dimz ; z++)
                for (int y = 0; y <  dimy; y++)
                    for (int x = 0; x < dimx; x++) {
                        if (*gt.data(i, 0, z, y, x)) {
                            counter += CHECK3();
                            //printf("i %d x %d y %d z %d in %d\n", i, x, y, z, res);
                        }
                    }
        return counter;
    }

    void extend3(py::array_t<bool> &in, py::array_t<bool> &out) {
        assert(in.ndim() == 5);
        assert(out.ndim() == 5);
        int n_batch = in.shape(0);
        assert(in.shape(1) == 1);
        int dimz = in.shape(2);
        int dimy = in.shape(3);
        int dimx = in.shape(4);

        auto& ptr = in;
        for (int i = 0; i < n_batch; i++) 
            for (int z = 1 ; z < dimz - 1; z++)
                for (int y = 1; y <  dimy - 1; y++)
                    for (int x = 1; x < dimx - 1; x++) {
                        *out.mutable_data(i, 0, z, y, x) = CHECK3();
                    }
    }

    void backproject_depth_ushort(py::array_t<unsigned short>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy, float normalizer) {
        assert(in.ndim() == 2);
        assert(out.ndim() == 3);

        int width = in.shape(1);
        int height = in.shape(0);
        assert(out.shape(0) == 3);
        assert(out.shape(1) == height);
        assert(out.shape(2) == width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth = float(*in.data(y, x)) / normalizer;

                if (depth > 0) {
                    float pos_x = depth * (x - cx) / fx;
                    float pos_y = depth * (y - cy) / fy;
                    float pos_z = depth;

                    *out.mutable_data(0, y, x) = pos_x;
                    *out.mutable_data(1, y, x) = pos_y;
                    *out.mutable_data(2, y, x) = pos_z;
                }
            }
        }
    }

    void backproject_depth_float(py::array_t<float>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy) {
        assert(in.ndim() == 2);
        assert(out.ndim() == 3);

        int width = in.shape(1);
        int height = in.shape(0);
        assert(out.shape(0) == 3);
        assert(out.shape(1) == height);
        assert(out.shape(2) == width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth = *in.data(y, x);

                if (depth > 0) {
                    float pos_x = depth * (x - cx) / fx;
                    float pos_y = depth * (y - cy) / fy;
                    float pos_z = depth;

                    *out.mutable_data(0, y, x) = pos_x;
                    *out.mutable_data(1, y, x) = pos_y;
                    *out.mutable_data(2, y, x) = pos_z;
                }
            }
        }
    }

    void compute_mesh_from_depth(
        const py::array_t<float>& pointImage, float maxTriangleEdgeDistance, 
        py::array_t<float>& vertexPositions, py::array_t<int>& faceIndices
    ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);

        // Compute valid pixel vertices and faces.
        // We also need to compute the pixel -> vertex index mapping for 
        // computation of faces.
        // We connect neighboring pixels on the square into two triangles.
        // We only select valid triangles, i.e. with all valid vertices and
        // not too far apart.
        // Important: The triangle orientation is set such that the normals
        // point towards the camera.
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;

        int vertexIdx = 0;
        std::vector<int> mapPixelToVertexIdx(width * height, -1);

        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                Eigen::Vector3f obs00(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                Eigen::Vector3f obs01(*pointImage.data(0, y + 1, x), *pointImage.data(1, y + 1, x), *pointImage.data(2, y + 1, x));
                Eigen::Vector3f obs10(*pointImage.data(0, y, x + 1), *pointImage.data(1, y, x + 1), *pointImage.data(2, y, x + 1));
                Eigen::Vector3f obs11(*pointImage.data(0, y + 1, x + 1), *pointImage.data(1, y + 1, x + 1), *pointImage.data(2, y + 1, x + 1));

                int idx00 = y * width + x;
                int idx01 = (y + 1) * width + x;
                int idx10 = y * width + (x + 1);
                int idx11 = (y + 1) * width + (x + 1);

                bool valid00 = obs00.z() > 0;
                bool valid01 = obs01.z() > 0;
                bool valid10 = obs10.z() > 0;
                bool valid11 = obs11.z() > 0;

                if (valid00 && valid01 && valid10) {
                    float d0 = (obs00 - obs01).norm();
                    float d1 = (obs00 - obs10).norm();
                    float d2 = (obs01 - obs10).norm();
                    
                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx00];
                        int vIdx1 = mapPixelToVertexIdx[idx01];
                        int vIdx2 = mapPixelToVertexIdx[idx10];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx00] = vertexIdx;
                            vertices.push_back(obs00);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }

                if (valid01 && valid10 && valid11) {
                    float d0 = (obs10 - obs01).norm();
                    float d1 = (obs10 - obs11).norm();
                    float d2 = (obs01 - obs11).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx11];
                        int vIdx1 = mapPixelToVertexIdx[idx10];
                        int vIdx2 = mapPixelToVertexIdx[idx01];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx11] = vertexIdx;
                            vertices.push_back(obs11);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }
            }
        }

        // Convert to numpy array.
        int nVertices = vertices.size();
        int nFaces = faces.size();

        if (nVertices > 0 && nFaces > 0) {
            // Reference check should be set to false otherwise there is a runtime
            // error. Check why that is the case.
            vertexPositions.resize({ nVertices, 3 }, false);
            faceIndices.resize({ nFaces, 3 }, false);

            for (int i = 0; i < nVertices; i++) {
                *vertexPositions.mutable_data(i, 0) = vertices[i].x();
                *vertexPositions.mutable_data(i, 1) = vertices[i].y();
                *vertexPositions.mutable_data(i, 2) = vertices[i].z();
            }
            
            for (int i = 0; i < nFaces; i++) {
                *faceIndices.mutable_data(i, 0) = faces[i].x();
                *faceIndices.mutable_data(i, 1) = faces[i].y();
                *faceIndices.mutable_data(i, 2) = faces[i].z();
            }
        }
    }

    void compute_mesh_from_depth_and_color(
        const py::array_t<float>& pointImage, const py::array_t<int>& colorImage, float maxTriangleEdgeDistance, 
        py::array_t<float>& vertexPositions, py::array_t<int>& vertexColors, py::array_t<int>& faceIndices
    ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);

        // Compute valid pixel vertices and faces.
        // We also need to compute the pixel -> vertex index mapping for 
        // computation of faces.
        // We connect neighboring pixels on the square into two triangles.
        // We only select valid triangles, i.e. with all valid vertices and
        // not too far apart.
        // Important: The triangle orientation is set such that the normals
        // point towards the camera.
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> colors;
        std::vector<Eigen::Vector3i> faces;

        int vertexIdx = 0;
        std::vector<int> mapPixelToVertexIdx(width * height, -1);

        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                Eigen::Vector3f obs00(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                Eigen::Vector3f obs01(*pointImage.data(0, y + 1, x), *pointImage.data(1, y + 1, x), *pointImage.data(2, y + 1, x));
                Eigen::Vector3f obs10(*pointImage.data(0, y, x + 1), *pointImage.data(1, y, x + 1), *pointImage.data(2, y, x + 1));
                Eigen::Vector3f obs11(*pointImage.data(0, y + 1, x + 1), *pointImage.data(1, y + 1, x + 1), *pointImage.data(2, y + 1, x + 1));

                Eigen::Vector3i color00(*colorImage.data(0, y, x), *colorImage.data(1, y, x), *colorImage.data(2, y, x));
                Eigen::Vector3i color01(*colorImage.data(0, y + 1, x), *colorImage.data(1, y + 1, x), *colorImage.data(2, y + 1, x));
                Eigen::Vector3i color10(*colorImage.data(0, y, x + 1), *colorImage.data(1, y, x + 1), *colorImage.data(2, y, x + 1));
                Eigen::Vector3i color11(*colorImage.data(0, y + 1, x + 1), *colorImage.data(1, y + 1, x + 1), *colorImage.data(2, y + 1, x + 1));

                int idx00 = y * width + x;
                int idx01 = (y + 1) * width + x;
                int idx10 = y * width + (x + 1);
                int idx11 = (y + 1) * width + (x + 1);

                bool valid00 = obs00.z() > 0;
                bool valid01 = obs01.z() > 0;
                bool valid10 = obs10.z() > 0;
                bool valid11 = obs11.z() > 0;

                if (valid00 && valid01 && valid10) {
                    float d0 = (obs00 - obs01).norm();
                    float d1 = (obs00 - obs10).norm();
                    float d2 = (obs01 - obs10).norm();
                    
                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx00];
                        int vIdx1 = mapPixelToVertexIdx[idx01];
                        int vIdx2 = mapPixelToVertexIdx[idx10];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx00] = vertexIdx;
                            vertices.push_back(obs00);
                            colors.push_back(color00);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            colors.push_back(color01);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            colors.push_back(color10);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }

                if (valid01 && valid10 && valid11) {
                    float d0 = (obs10 - obs01).norm();
                    float d1 = (obs10 - obs11).norm();
                    float d2 = (obs01 - obs11).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx11];
                        int vIdx1 = mapPixelToVertexIdx[idx10];
                        int vIdx2 = mapPixelToVertexIdx[idx01];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx11] = vertexIdx;
                            vertices.push_back(obs11);
                            colors.push_back(color11);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            colors.push_back(color10);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            colors.push_back(color01);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }
            }
        }

        // Convert to numpy array.
        int nVertices = vertices.size();
        int nFaces = faces.size();

        if (nVertices > 0 && nFaces > 0) {
            // Reference check should be set to false otherwise there is a runtime
            // error. Check why that is the case.
            vertexPositions.resize({ nVertices, 3 }, false);
            vertexColors.resize({ nVertices, 3 }, false);
            faceIndices.resize({ nFaces, 3 }, false);

            for (int i = 0; i < nVertices; i++) {
                *vertexPositions.mutable_data(i, 0) = vertices[i].x();
                *vertexPositions.mutable_data(i, 1) = vertices[i].y();
                *vertexPositions.mutable_data(i, 2) = vertices[i].z();

                *vertexColors.mutable_data(i, 0) = colors[i].x();
                *vertexColors.mutable_data(i, 1) = colors[i].y();
                *vertexColors.mutable_data(i, 2) = colors[i].z();
            }
            
            for (int i = 0; i < nFaces; i++) {
                *faceIndices.mutable_data(i, 0) = faces[i].x();
                *faceIndices.mutable_data(i, 1) = faces[i].y();
                *faceIndices.mutable_data(i, 2) = faces[i].z();
            }
        }
    }

    void compute_mesh_from_depth_and_flow(
        const py::array_t<float>& pointImage, const py::array_t<float>& flowImage, float maxTriangleEdgeDistance, 
        py::array_t<float>& vertexPositions, py::array_t<float>& vertexFlows, py::array_t<int>& vertexPixels, py::array_t<int>& faceIndices
    ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);

        // Compute valid pixel vertices and faces.
        // We also need to compute the pixel -> vertex index mapping for 
        // computation of faces.
        // We connect neighboring pixels on the square into two triangles.
        // We only select valid triangles, i.e. with all valid vertices and
        // not too far apart.
        // Important: The triangle orientation is set such that the normals
        // point towards the camera.
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3f> flows;
        std::vector<Eigen::Vector2i> pixels;
        std::vector<Eigen::Vector3i> faces;

        int vertexIdx = 0;
        std::vector<int> mapPixelToVertexIdx(width * height, -1);

        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                Eigen::Vector3f obs00(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                Eigen::Vector3f obs01(*pointImage.data(0, y + 1, x), *pointImage.data(1, y + 1, x), *pointImage.data(2, y + 1, x));
                Eigen::Vector3f obs10(*pointImage.data(0, y, x + 1), *pointImage.data(1, y, x + 1), *pointImage.data(2, y, x + 1));
                Eigen::Vector3f obs11(*pointImage.data(0, y + 1, x + 1), *pointImage.data(1, y + 1, x + 1), *pointImage.data(2, y + 1, x + 1));

                Eigen::Vector3f flow00(*flowImage.data(0, y, x), *flowImage.data(1, y, x), *flowImage.data(2, y, x));
                Eigen::Vector3f flow01(*flowImage.data(0, y + 1, x), *flowImage.data(1, y + 1, x), *flowImage.data(2, y + 1, x));
                Eigen::Vector3f flow10(*flowImage.data(0, y, x + 1), *flowImage.data(1, y, x + 1), *flowImage.data(2, y, x + 1));
                Eigen::Vector3f flow11(*flowImage.data(0, y + 1, x + 1), *flowImage.data(1, y + 1, x + 1), *flowImage.data(2, y + 1, x + 1));

                int idx00 = y * width + x;
                int idx01 = (y + 1) * width + x;
                int idx10 = y * width + (x + 1);
                int idx11 = (y + 1) * width + (x + 1);

                bool valid00 = obs00.z() > 0 && std::isfinite(flow00.x()) && std::isfinite(flow00.y()) && std::isfinite(flow00.z());
                bool valid01 = obs01.z() > 0 && std::isfinite(flow01.x()) && std::isfinite(flow01.y()) && std::isfinite(flow01.z());
                bool valid10 = obs10.z() > 0 && std::isfinite(flow10.x()) && std::isfinite(flow10.y()) && std::isfinite(flow10.z());
                bool valid11 = obs11.z() > 0 && std::isfinite(flow11.x()) && std::isfinite(flow11.y()) && std::isfinite(flow11.z());

                if (valid00 && valid01 && valid10) {
                    float d0 = (obs00 - obs01).norm();
                    float d1 = (obs00 - obs10).norm();
                    float d2 = (obs01 - obs10).norm();
                    
                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx00];
                        int vIdx1 = mapPixelToVertexIdx[idx01];
                        int vIdx2 = mapPixelToVertexIdx[idx10];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx00] = vertexIdx;
                            vertices.push_back(obs00);
                            flows.push_back(flow00);
                            pixels.push_back(Eigen::Vector2i(x, y));
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            flows.push_back(flow01);
                            pixels.push_back(Eigen::Vector2i(x, y + 1));
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            flows.push_back(flow10);
                            pixels.push_back(Eigen::Vector2i(x + 1, y));
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }

                if (valid01 && valid10 && valid11) {
                    float d0 = (obs10 - obs01).norm();
                    float d1 = (obs10 - obs11).norm();
                    float d2 = (obs01 - obs11).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx11];
                        int vIdx1 = mapPixelToVertexIdx[idx10];
                        int vIdx2 = mapPixelToVertexIdx[idx01];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx11] = vertexIdx;
                            vertices.push_back(obs11);
                            flows.push_back(flow11);
                            pixels.push_back(Eigen::Vector2i(x + 1, y + 1));
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            flows.push_back(flow10);
                            pixels.push_back(Eigen::Vector2i(x + 1, y));
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            flows.push_back(flow01);
                            pixels.push_back(Eigen::Vector2i(x, y + 1));
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }
            }
        }

        // Convert to numpy array.
        int nVertices = vertices.size();
        int nFaces = faces.size();

        if (nVertices > 0 && nFaces > 0) {
            // Reference check should be set to false otherwise there is a runtime
            // error. Check why that is the case.
            vertexPositions.resize({ nVertices, 3 }, false);
            vertexFlows.resize({ nVertices, 3 }, false);
            vertexPixels.resize({ nVertices, 2 }, false);
            faceIndices.resize({ nFaces, 3 }, false);

            for (int i = 0; i < nVertices; i++) {
                *vertexPositions.mutable_data(i, 0) = vertices[i].x();
                *vertexPositions.mutable_data(i, 1) = vertices[i].y();
                *vertexPositions.mutable_data(i, 2) = vertices[i].z();

                *vertexFlows.mutable_data(i, 0) = flows[i].x();
                *vertexFlows.mutable_data(i, 1) = flows[i].y();
                *vertexFlows.mutable_data(i, 2) = flows[i].z();

                *vertexPixels.mutable_data(i, 0) = pixels[i].x();
                *vertexPixels.mutable_data(i, 1) = pixels[i].y();
            }
            
            for (int i = 0; i < nFaces; i++) {
                *faceIndices.mutable_data(i, 0) = faces[i].x();
                *faceIndices.mutable_data(i, 1) = faces[i].y();
                *faceIndices.mutable_data(i, 2) = faces[i].z();
            }
        }
    }

    void filter_depth(py::array_t<unsigned short>& in, py::array_t<unsigned short>& out, int radius) {
        assert(in.ndim() == 2);
        assert(out.ndim() == 2);
        
        unsigned kernelSize = 2 * radius + 1;
        unsigned windowSize = kernelSize * kernelSize;

        int width = in.shape(1);
        int height = in.shape(0);
        assert(out.shape(0) == height);
        assert(out.shape(1) == width);

        // #pragma omp parallel for
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Get all values in the median window.
                int xMin = std::max(x - radius, 0);
                int xMax = std::min(x + radius, int(width) - 1);
                int yMin = std::max(y - radius, 0);
                int yMax = std::min(y + radius, int(height) - 1);

                std::vector<unsigned short> windowValues;
                windowValues.reserve(windowSize);

                for (int yNear = yMin; yNear <= yMax; yNear++) {
                    for (int xNear = xMin; xNear <= xMax; xNear++) {
                        unsigned short depth = *in.data(yNear, xNear);
                        if (depth > 0) {
                            windowValues.push_back(depth);
                        }
                    }
                }

                // Sort the values and pick the median as the middle element.
                unsigned nElements = windowValues.size();
                std::sort(windowValues.begin(), windowValues.end());

                unsigned middleIdx = std::floor(nElements / 2);
                unsigned short median = windowValues[middleIdx];

                // Write out the median value.
                *out.mutable_data(y, x) = median;
            }
        }
    }

    py::array_t<float> warp_flow(const py::array_t<float>& image, const py::array_t<float>& flow, const py::array_t<float>& mask) {
        // We assume:
        //      image shape (3, h, w)
        //      flow shape  (2, h, w)
        //      mask shape  (2, h, w)

        int width = image.shape(2);
        int height = image.shape(1);

        py::array_t<float> imageWarped = py::array_t<float>({3, height, width});
        py::array_t<float> weightsWarped = py::array_t<float>({1, height, width});

        // Initialize to zero.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                *imageWarped.mutable_data(0, v, u) = 0.0;
                *imageWarped.mutable_data(1, v, u) = 0.0;
                *imageWarped.mutable_data(2, v, u) = 0.0;
                *weightsWarped.mutable_data(0, v, u) = 0.0;
            }
        }

        // Compute image values and interpolation weights.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                // Check if pixel is inside the mask.
                if (*mask.data(0, v, u) <= 0 || *mask.data(1, v, u) <= 0) continue;

                // Compute the warped pixel.
                float u_warped = u + *flow.data(0, v, u);
                float v_warped = v + *flow.data(1, v, u);

                int u0 = std::floor(u_warped);
                int u1 = u0 + 1;
                int v0 = std::floor(v_warped);
                int v1 = v0 + 1;

                if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

                // Interpolate the color contributions.
                float du = u_warped - u0;
                float dv = v_warped - v0;
                
                float w00 = (1 - du)*(1 - dv);
                float w01 = (1 - du)*dv;
                float w10 = du*(1 - dv);
                float w11 = du*dv;

                float c0 = *image.data(0, v, u); 
                float c1 = *image.data(1, v, u); 
                float c2 = *image.data(2, v, u); 

                *imageWarped.mutable_data(0, v0, u0) += w00 * c0;
                *imageWarped.mutable_data(1, v0, u0) += w00 * c1;
                *imageWarped.mutable_data(2, v0, u0) += w00 * c2;
                *imageWarped.mutable_data(0, v1, u0) += w01 * c0;
                *imageWarped.mutable_data(1, v1, u0) += w01 * c1;
                *imageWarped.mutable_data(2, v1, u0) += w01 * c2;
                *imageWarped.mutable_data(0, v0, u1) += w10 * c0;
                *imageWarped.mutable_data(1, v0, u1) += w10 * c1;
                *imageWarped.mutable_data(2, v0, u1) += w10 * c2;
                *imageWarped.mutable_data(0, v1, u1) += w11 * c0;
                *imageWarped.mutable_data(1, v1, u1) += w11 * c1;
                *imageWarped.mutable_data(2, v1, u1) += w11 * c2;
                
                *weightsWarped.mutable_data(0, v0, u0) += w00;
                *weightsWarped.mutable_data(0, v1, u0) += w01;
                *weightsWarped.mutable_data(0, v0, u1) += w10;
                *weightsWarped.mutable_data(0, v1, u1) += w11;
            }
        }

        // Normalize image.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                float w = *weightsWarped.data(0, v, u);
                if (w > 0) {
                    *imageWarped.mutable_data(0, v, u) /= w;
                    *imageWarped.mutable_data(1, v, u) /= w;
                    *imageWarped.mutable_data(2, v, u) /= w;
                }
                else {
                    *imageWarped.mutable_data(0, v, u) = 1.0;
                    *imageWarped.mutable_data(1, v, u) = 1.0;
                    *imageWarped.mutable_data(2, v, u) = 1.0;
                }
            }
        }

        return imageWarped;
    }

    py::array_t<float> warp_rigid(
        const py::array_t<float>& rgbd, 
        const py::array_t<float>& rotation, 
        const py::array_t<float>& translation, 
        float fx, float fy, float cx, float cy
    ) { 
        // We assume:
        //      rgbd shape (6, h, w)
        //      rotation shape  (9)
        //      translation shape  (2)

        int width = rgbd.shape(2);
        int height = rgbd.shape(1);

        float r00 = *rotation.data(0);
        float r01 = *rotation.data(1);
        float r02 = *rotation.data(2);
        float r10 = *rotation.data(3);
        float r11 = *rotation.data(4);
        float r12 = *rotation.data(5);
        float r20 = *rotation.data(6);
        float r21 = *rotation.data(7);
        float r22 = *rotation.data(8);
        float t0 = *translation.data(0);
        float t1 = *translation.data(1);
        float t2 = *translation.data(2);

        py::array_t<float> imageWarped = py::array_t<float>({3, height, width});
        py::array_t<float> weightsWarped = py::array_t<float>({1, height, width});

        // Initialize to zero.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                *imageWarped.mutable_data(0, v, u) = 0.0;
                *imageWarped.mutable_data(1, v, u) = 0.0;
                *imageWarped.mutable_data(2, v, u) = 0.0;
                *weightsWarped.mutable_data(0, v, u) = 0.0;
            }
        }

        // Compute image values and interpolation weights.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                // Compute the warped pixel.
                float x = *rgbd.data(3, v, u);
                float y = *rgbd.data(4, v, u);
                float z = *rgbd.data(5, v, u);
                if (z <= 0) continue;

                float x_def = r00 * x + r01 * y + r02 * z + t0;
                float y_def = r10 * x + r11 * y + r12 * z + t1;
                float z_def = r20 * x + r21 * y + r22 * z + t2;
                if (z_def <= 0) continue;

                float u_warped = fx * x_def / z_def + cx;
                float v_warped = fy * y_def / z_def + cy;

                int u0 = std::floor(u_warped);
                int u1 = u0 + 1;
                int v0 = std::floor(v_warped);
                int v1 = v0 + 1;

                if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

                // Interpolate the color contributions.
                float du = u_warped - u0;
                float dv = v_warped - v0;
                
                float w00 = (1 - du)*(1 - dv);
                float w01 = (1 - du)*dv;
                float w10 = du*(1 - dv);
                float w11 = du*dv;

                float c0 = *rgbd.data(0, v, u); 
                float c1 = *rgbd.data(1, v, u); 
                float c2 = *rgbd.data(2, v, u); 

                *imageWarped.mutable_data(0, v0, u0) += w00 * c0;
                *imageWarped.mutable_data(1, v0, u0) += w00 * c1;
                *imageWarped.mutable_data(2, v0, u0) += w00 * c2;
                *imageWarped.mutable_data(0, v1, u0) += w01 * c0;
                *imageWarped.mutable_data(1, v1, u0) += w01 * c1;
                *imageWarped.mutable_data(2, v1, u0) += w01 * c2;
                *imageWarped.mutable_data(0, v0, u1) += w10 * c0;
                *imageWarped.mutable_data(1, v0, u1) += w10 * c1;
                *imageWarped.mutable_data(2, v0, u1) += w10 * c2;
                *imageWarped.mutable_data(0, v1, u1) += w11 * c0;
                *imageWarped.mutable_data(1, v1, u1) += w11 * c1;
                *imageWarped.mutable_data(2, v1, u1) += w11 * c2;
                
                *weightsWarped.mutable_data(0, v0, u0) += w00;
                *weightsWarped.mutable_data(0, v1, u0) += w01;
                *weightsWarped.mutable_data(0, v0, u1) += w10;
                *weightsWarped.mutable_data(0, v1, u1) += w11;
            }
        }

        // Normalize image.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                float w = *weightsWarped.data(0, v, u);
                if (w > 0) {
                    *imageWarped.mutable_data(0, v, u) /= w;
                    *imageWarped.mutable_data(1, v, u) /= w;
                    *imageWarped.mutable_data(2, v, u) /= w;
                }
                else {
                    *imageWarped.mutable_data(0, v, u) = 1.0;
                    *imageWarped.mutable_data(1, v, u) = 1.0;
                    *imageWarped.mutable_data(2, v, u) = 1.0;
                }
            }
        }

        return imageWarped;
    }

    py::array_t<float> warp_3d(const py::array_t<float>& rgbd, const py::array_t<float>& points, const py::array_t<int>& pointValidity, float fx, float fy, float cx, float cy) {
        // We assume:
        //      image shape             (6, h, w)
        //      points shape            (3, h, w)
        //      pointValidity shape     (h, w)

        int width = rgbd.shape(2);
        int height = rgbd.shape(1);

        py::array_t<float> imageWarped = py::array_t<float>({3, height, width});
        py::array_t<float> weightsWarped = py::array_t<float>({1, height, width});

        // Initialize to zero.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                *imageWarped.mutable_data(0, v, u) = 0.0;
                *imageWarped.mutable_data(1, v, u) = 0.0;
                *imageWarped.mutable_data(2, v, u) = 0.0;
                *weightsWarped.mutable_data(0, v, u) = 0.0;
            }
        }

        // Compute image values and interpolation weights.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                // Compute the warped pixel.
                if (*pointValidity.data(v, u) <= 0) continue;

                float z = *rgbd.data(5, v, u);
                if (z <= 0) continue;

                float x_def = *points.data(0, v, u);
                float y_def = *points.data(1, v, u);
                float z_def = *points.data(2, v, u);
                if (z_def <= 0) continue;

                float u_warped = fx * x_def / z_def + cx;
                float v_warped = fy * y_def / z_def + cy;

                int u0 = std::floor(u_warped);
                int u1 = u0 + 1;
                int v0 = std::floor(v_warped);
                int v1 = v0 + 1;

                if (u0 < 0 || u1 >= width || v0 < 0 || v1 >= height) continue;

                // Interpolate the color contributions.
                float du = u_warped - u0;
                float dv = v_warped - v0;
                
                float w00 = (1 - du)*(1 - dv);
                float w01 = (1 - du)*dv;
                float w10 = du*(1 - dv);
                float w11 = du*dv;

                float c0 = *rgbd.data(0, v, u); 
                float c1 = *rgbd.data(1, v, u); 
                float c2 = *rgbd.data(2, v, u); 

                *imageWarped.mutable_data(0, v0, u0) += w00 * c0;
                *imageWarped.mutable_data(1, v0, u0) += w00 * c1;
                *imageWarped.mutable_data(2, v0, u0) += w00 * c2;
                *imageWarped.mutable_data(0, v1, u0) += w01 * c0;
                *imageWarped.mutable_data(1, v1, u0) += w01 * c1;
                *imageWarped.mutable_data(2, v1, u0) += w01 * c2;
                *imageWarped.mutable_data(0, v0, u1) += w10 * c0;
                *imageWarped.mutable_data(1, v0, u1) += w10 * c1;
                *imageWarped.mutable_data(2, v0, u1) += w10 * c2;
                *imageWarped.mutable_data(0, v1, u1) += w11 * c0;
                *imageWarped.mutable_data(1, v1, u1) += w11 * c1;
                *imageWarped.mutable_data(2, v1, u1) += w11 * c2;
                
                *weightsWarped.mutable_data(0, v0, u0) += w00;
                *weightsWarped.mutable_data(0, v1, u0) += w01;
                *weightsWarped.mutable_data(0, v0, u1) += w10;
                *weightsWarped.mutable_data(0, v1, u1) += w11;
            }
        }

        // Normalize image.
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                float w = *weightsWarped.data(0, v, u);
                if (w > 0) {
                    *imageWarped.mutable_data(0, v, u) /= w;
                    *imageWarped.mutable_data(1, v, u) /= w;
                    *imageWarped.mutable_data(2, v, u) /= w;
                }
                else {
                    *imageWarped.mutable_data(0, v, u) = 1.0;
                    *imageWarped.mutable_data(1, v, u) = 1.0;
                    *imageWarped.mutable_data(2, v, u) = 1.0;
                }
            }
        }

        return imageWarped;
    }

} //namespace image_proc