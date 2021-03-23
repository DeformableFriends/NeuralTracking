import os
import shutil
import numpy as np
import json
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
import datetime
import open3d as o3d
import trimesh
import re

from utils import utils, image_proc

from NeuralNRT._C import compute_mesh_from_depth_and_flow as compute_mesh_from_depth_and_flow_c
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import node_and_edge_clean_up as node_and_edge_clean_up_c
from NeuralNRT._C import compute_pixel_anchors_geodesic as compute_pixel_anchors_geodesic_c
from NeuralNRT._C import compute_clusters as compute_clusters_c
from NeuralNRT._C import update_pixel_anchors as update_pixel_anchors_c


if __name__ == "__main__":
    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion
    DEPTH_NORMALIZER = 1000.0
    MAX_TRIANGLE_DISTANCE = 0.05

    # Erosion of vertices in the boundaries
    EROSION_NUM_ITERATIONS = 10
    EROSION_MIN_NEIGHBORS = 4

    # Node sampling and edges computation
    NODE_COVERAGE = 0.05 # in meters
    USE_ONLY_VALID_VERTICES = True
    NUM_NEIGHBORS = 8
    ENFORCE_TOTAL_NUM_NEIGHBORS = False
    SAMPLE_RANDOM_SHUFFLE = False

    # Pixel anchors
    NEIGHBORHOOD_DEPTH = 2

    MIN_CLUSTER_SIZE = 3
    MIN_NUM_NEIGHBORS = 2 

    # Node clean-up
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True

    #########################################################################
    # Paths.
    #########################################################################
    seq_dir = os.path.join("example_data" , "train", "seq258")

    depth_image_path = os.path.join(seq_dir, "depth", "000000.png")
    mask_image_path = os.path.join(seq_dir, "mask", "000000_shirt.png")
    scene_flow_path = os.path.join(seq_dir, "scene_flow", "shirt_000000_000110.sflow")
    intrinsics_path = os.path.join(seq_dir, "intrinsics.txt")

    pair_name = "generated_shirt_000000_000110"

    #########################################################################
    # Load data.
    #########################################################################
    # Load intrinsics.
    intrinsics = np.loadtxt(intrinsics_path)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Load depth image.
    depth_image = io.imread(depth_image_path) 

    # Load mask image.
    mask_image = io.imread(mask_image_path) 

    # Load scene flow image.
    scene_flow_image = utils.load_flow(scene_flow_path)

    #########################################################################
    # Convert depth to mesh.
    #########################################################################
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    # Invalidate depth values outside object mask.
    # We only define graph over dynamic object (inside the object mask).
    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    # Backproject depth images into 3D.
    point_image = image_proc.backproject_depth(depth_image, fx, fy, cx, cy, normalizer=DEPTH_NORMALIZER)
    point_image = point_image.astype(np.float32)

    # Convert depth image into mesh, using pixelwise connectivity.
    # We also compute flow values, and invalidate any vertex with non-finite
    # flow values.
    vertices = np.zeros((0), dtype=np.float32)
    vertex_flows = np.zeros((0), dtype=np.float32)
    vertex_pixels = np.zeros((0), dtype=np.int32)
    faces = np.zeros((0), dtype=np.int32)

    compute_mesh_from_depth_and_flow_c(
        point_image, scene_flow_image, 
        MAX_TRIANGLE_DISTANCE, 
        vertices, vertex_flows, vertex_pixels, faces
    )
    
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    assert num_vertices > 0 and num_faces > 0

    # Erode mesh, to not sample unstable nodes on the mesh boundary.
    non_eroded_vertices = erode_mesh_c(
        vertices, faces, EROSION_NUM_ITERATIONS, EROSION_MIN_NEIGHBORS
    )

    # Just for debugging.
    # mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    # mesh.compute_vertex_normals()
    
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[non_eroded_vertices.reshape(-1), :]))

    # o3d.visualization.draw_geometries([mesh, pcd], mesh_show_back_face=True)

    # mesh_transformed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices + vertex_flows), o3d.utility.Vector3iVector(faces))
    # mesh_transformed.compute_vertex_normals()
    # mesh_transformed.paint_uniform_color([0.0, 1.0, 0.0])

    # o3d.visualization.draw_geometries([mesh, mesh_transformed], mesh_show_back_face=True)

    #########################################################################
    # Sample graph nodes.
    #########################################################################
    valid_vertices = non_eroded_vertices

    # Sample graph nodes.
    node_coords = np.zeros((0), dtype=np.float32)
    node_indices = np.zeros((0), dtype=np.int32)

    num_nodes = sample_nodes_c(
        vertices, valid_vertices,
        node_coords, node_indices, 
        NODE_COVERAGE, 
        USE_ONLY_VALID_VERTICES,
        SAMPLE_RANDOM_SHUFFLE
    )

    node_coords = node_coords[:num_nodes, :]
    node_indices = node_indices[:num_nodes, :]

    # Get node deformation.
    node_deformations = vertex_flows[node_indices.squeeze()]
    node_deformations = node_deformations.reshape(-1, 3)

    assert np.isfinite(node_deformations).all(), "All deformations should be valid."
    assert node_deformations.shape[0] == node_coords.shape[0] == node_indices.shape[0]

    # Just for debugging
    # pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
    # o3d.visualization.draw_geometries([pcd_nodes], mesh_show_back_face=True)

    #########################################################################
    # Compute graph edges.
    #########################################################################
    # Compute edges between nodes.
    graph_edges              = -np.ones((num_nodes, NUM_NEIGHBORS), dtype=np.int32)
    graph_edges_weights      =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
    graph_edges_distances    =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
    node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

    visible_vertices = np.ones_like(valid_vertices)

    compute_edges_geodesic_c(
        vertices, visible_vertices, faces, node_indices, 
        NUM_NEIGHBORS, NODE_COVERAGE, 
        graph_edges, graph_edges_weights, graph_edges_distances,
        node_to_vertex_distances,
        USE_ONLY_VALID_VERTICES,
        ENFORCE_TOTAL_NUM_NEIGHBORS
    )

    # Remove nodes 
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
    node_id_black_list = []

    if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
        # Mark nodes with not enough neighbors
        node_and_edge_clean_up_c(graph_edges, valid_nodes_mask)

        # Get the list of invalid nodes
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    else:
        print("You're allowing nodes with not enough neighbors!")

    print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(node_id_black_list), "({})".format(node_id_black_list))

    #########################################################################
    # Compute pixel anchors.
    #########################################################################
    pixel_anchors = np.zeros((0), dtype=np.int32)
    pixel_weights = np.zeros((0), dtype=np.float32)

    compute_pixel_anchors_geodesic_c(
        node_to_vertex_distances, valid_nodes_mask, 
        vertices, vertex_pixels, 
        pixel_anchors, pixel_weights,
        width, height, NODE_COVERAGE
    )
   
    print("Valid pixels:", np.sum(np.all(pixel_anchors != -1, axis=2)))

    # Just for debugging.
    # pixel_anchors_image = np.sum(pixel_anchors, axis=2)
    # pixel_anchors_mask = np.copy(pixel_anchors_image).astype(np.uint8)
    # pixel_anchors_mask[...] = 1
    # pixel_anchors_mask[pixel_anchors_image == -4] = 0
    # utils.save_grayscale_image("output/pixel_anchors_mask.jpeg", pixel_anchors_mask)

    # Get only valid nodes and their corresponding info
    node_coords           = node_coords[valid_nodes_mask.squeeze()]
    node_indices          = node_indices[valid_nodes_mask.squeeze()]
    node_deformations     = node_deformations[valid_nodes_mask.squeeze()]
    graph_edges           = graph_edges[valid_nodes_mask.squeeze()] 
    graph_edges_weights   = graph_edges_weights[valid_nodes_mask.squeeze()] 
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()] 

    #########################################################################
    # Graph checks.
    #########################################################################
    num_nodes = node_coords.shape[0]

    # Check that we have enough nodes
    if (num_nodes == 0):
        print("No nodes! Exiting ...")
        exit()

    # Update node ids only if we actually removed nodes
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy           = np.copy(graph_edge)
            graph_edge_weights_copy   = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id]           = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id]   =  np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] =  np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]    

                    # get mapped neighbor id       
                    if current_neighbor_id == -1: mapped_neighbor_id = -1
                    else:                         mapped_neighbor_id = node_id_mapping[current_neighbor_id]    

                    graph_edges[node_id, count_valid_neighbors]           = mapped_neighbor_id
                    graph_edges_weights[node_id, count_valid_neighbors]   = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                print("Hmmmmm", graph_edges_weights[node_id])
                raise Exception("Not good")

        # 3. Update pixel anchors using the id mapping (note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_geodesic_c' was given 'valid_nodes_mask')
        update_pixel_anchors_c(node_id_mapping, pixel_anchors)

    #########################################################################
    # Compute clusters.
    #########################################################################
    graph_clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32)
    clusters_size_list = compute_clusters_c(graph_edges, graph_clusters)

    for i, cluster_size in enumerate(clusters_size_list):
        if cluster_size <= 2:
            print("Cluster is too small {}".format(clusters_size_list))
            print("It only has nodes:", np.where(graph_clusters == i)[0])
            exit()

    #########################################################################
    # Save data.
    #########################################################################
    dst_graph_nodes_dir = os.path.join(seq_dir, "graph_nodes")
    if not os.path.exists(dst_graph_nodes_dir): os.makedirs(dst_graph_nodes_dir)

    dst_graph_edges_dir = os.path.join(seq_dir, "graph_edges")
    if not os.path.exists(dst_graph_edges_dir): os.makedirs(dst_graph_edges_dir)

    dst_graph_edges_weights_dir = os.path.join(seq_dir, "graph_edges_weights")
    if not os.path.exists(dst_graph_edges_weights_dir): os.makedirs(dst_graph_edges_weights_dir)

    dst_node_deformations_dir = os.path.join(seq_dir, "graph_node_deformations")
    if not os.path.exists(dst_node_deformations_dir): os.makedirs(dst_node_deformations_dir)

    dst_graph_clusters_dir = os.path.join(seq_dir, "graph_clusters")
    if not os.path.exists(dst_graph_clusters_dir): os.makedirs(dst_graph_clusters_dir)

    dst_pixel_anchors_dir = os.path.join(seq_dir, "pixel_anchors")
    if not os.path.exists(dst_pixel_anchors_dir): os.makedirs(dst_pixel_anchors_dir)

    dst_pixel_weights_dir = os.path.join(seq_dir, "pixel_weights")
    if not os.path.exists(dst_pixel_weights_dir): os.makedirs(dst_pixel_weights_dir)

    output_graph_nodes_path           = os.path.join(dst_graph_nodes_dir, pair_name           + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_graph_edges_path           = os.path.join(dst_graph_edges_dir, pair_name           + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_graph_edges_weights_path   = os.path.join(dst_graph_edges_weights_dir, pair_name   + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_node_deformations_path     = os.path.join(dst_node_deformations_dir, pair_name     + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_graph_clusters_path        = os.path.join(dst_graph_clusters_dir, pair_name        + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_pixel_anchors_path         = os.path.join(dst_pixel_anchors_dir, pair_name         + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    output_pixel_weights_path         = os.path.join(dst_pixel_weights_dir, pair_name         + "_{}_{:.2f}.bin".format("geodesic", NODE_COVERAGE))
    
    utils.save_graph_nodes(output_graph_nodes_path, node_coords)
    utils.save_graph_edges(output_graph_edges_path, graph_edges)
    utils.save_graph_edges_weights(output_graph_edges_weights_path, graph_edges_weights)
    utils.save_graph_node_deformations(output_node_deformations_path, node_deformations)
    utils.save_graph_clusters(output_graph_clusters_path, graph_clusters)
    utils.save_int_image(output_pixel_anchors_path, pixel_anchors)
    utils.save_float_image(output_pixel_weights_path, pixel_weights)

    assert np.array_equal(node_coords, utils.load_graph_nodes(output_graph_nodes_path))
    assert np.array_equal(graph_edges, utils.load_graph_edges(output_graph_edges_path))
    assert np.array_equal(graph_edges_weights, utils.load_graph_edges_weights(output_graph_edges_weights_path))
    assert np.allclose(node_deformations, utils.load_graph_node_deformations(output_node_deformations_path))
    assert np.array_equal(graph_clusters, utils.load_graph_clusters(output_graph_clusters_path))
    assert np.array_equal(pixel_anchors, utils.load_int_image(output_pixel_anchors_path))
    assert np.array_equal(pixel_weights, utils.load_float_image(output_pixel_weights_path))

