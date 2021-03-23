#include <torch/extension.h>

#include "cpu/image_proc.h"
#include "cpu/graph_proc.h"

// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("compute_augmented_flow_from_rotation", 
        &image_proc::compute_augmented_flow_from_rotation, 
        "Computes an optical flow image that reflects the augmentation applied to the source and target images.");
  
  m.def("count_tp1", &image_proc::count_tp1, "");
  m.def("count_tp2", &image_proc::count_tp2, "");
  m.def("count_tp3", &image_proc::count_tp3, "");

  m.def("extend3", &image_proc::extend3, "");

  m.def("backproject_depth_ushort", &image_proc::backproject_depth_ushort, "Backproject depth image into 3D points");
  m.def("backproject_depth_float", &image_proc::backproject_depth_float, "Backproject depth image into 3D points");
  m.def("compute_mesh_from_depth", &image_proc::compute_mesh_from_depth, "Computes a mesh using backprojected points and pixel connectivity");
  m.def("compute_mesh_from_depth_and_color", &image_proc::compute_mesh_from_depth_and_color, "Computes a mesh using backprojected points and pixel connectivity. Additionally, extracts colors for each vertex");
  m.def("compute_mesh_from_depth_and_flow", &image_proc::compute_mesh_from_depth_and_flow, "Computes a mesh using backprojected points and pixel connectivity. Additionally, extracts flows for each vertex");
  m.def("filter_depth", &image_proc::filter_depth, "Executes median filter on depth image");

  m.def("warp_flow", &image_proc::warp_flow, "Warps image using provided 2D flow (inside masked region)");
  m.def("warp_rigid", &image_proc::warp_rigid, "Warps image using provided depth map and rigid pose");
  m.def("warp_3d", &image_proc::warp_3d, "Warps image using provided warped point cloud");

  m.def("erode_mesh", &graph_proc::erode_mesh, "Erode mesh");
  m.def("sample_nodes", &graph_proc::sample_nodes, "Samples graph nodes that cover given vertices");
  m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "Computes geodesic edges between given graph nodes");
  m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "Computes Euclidean edges between given graph nodes");
  m.def("node_and_edge_clean_up", &graph_proc::node_and_edge_clean_up, "Removes invalid nodes");
  m.def("compute_clusters", &graph_proc::compute_clusters, "Computes graph node clusters");
  m.def("compute_pixel_anchors_geodesic", &graph_proc::compute_pixel_anchors_geodesic, "Computes anchor ids and skinning weights for every pixel using graph connectivity");
  m.def("compute_pixel_anchors_euclidean", &graph_proc::compute_pixel_anchors_euclidean, "Computes anchor ids and skinning weights for every pixel using Euclidean distances");
  m.def("update_pixel_anchors", &graph_proc::update_pixel_anchors, "Updates pixel anchor after node id change");
  m.def("construct_regular_graph", &graph_proc::construct_regular_graph, "Samples graph uniformly in pixel space, and computes pixel anchors");
}