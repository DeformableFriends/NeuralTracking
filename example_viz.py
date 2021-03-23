import os

import open3d as o3d
import torch
import numpy as np

from utils import image_proc
from model.model import DeformNet
from model import dataset

import utils.utils as utils
import utils.viz_utils as viz_utils
import utils.nnutils as nnutils
import utils.line_mesh as line_mesh_utils
import options as opt


def main():

    #####################################################################################################
    # Options
    #####################################################################################################

    # Source-target example

    split = "test"
    seq_id = 17

    src_id = 300 # source frame
    tgt_id = 600 # target frame

    srt_tgt_str = "5dbd7c9104df0300f329f294_shirt_000300_000600_geodesic_0.05"

    # Make sure to use the intrinsics corresponding to the seq_id above!!  
    intrinsics = {
        "fx": 575.548,
        "fy": 577.46,
        "cx": 323.172,
        "cy": 236.417
    }

    # Train set example
    # Important: You need to generate graph data using create_graph_data.py first.

    # split = "train"
    # seq_id = 258

    # src_id = 0 # source frame
    # tgt_id = 110 # target frame

    # srt_tgt_str = "generated_shirt_000000_000110_geodesic_0.05"

    # # Make sure to use the intrinsics corresponding to the seq_id above!!  
    # intrinsics = {
    #     "fx": 575.548,
    #     "fy": 577.46,
    #     "cx": 323.172,
    #     "cy": 236.417
    # }

    # Some params for coloring the predicted correspondence confidences
    weight_thr = 0.3
    weight_scale = 1

    # We will overwrite the default value in options.py / settings.py
    opt.use_mask = True
    
    #####################################################################################################
    # Load model
    #####################################################################################################

    saved_model = opt.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct model
    model = DeformNet().cuda()

    if "chairs_things" in saved_model:
        model.flow_net.load_state_dict(pretrained_dict)
    else:
        if opt.model_module_to_load == "full_model":
            # Load completely model            
            model.load_state_dict(pretrained_dict)
        elif opt.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            print(opt.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    model.eval()

    #####################################################################################################
    # Load example dataset
    #####################################################################################################
    example_dir = os.path.join("example_data" , f"{split}/seq{str(seq_id).zfill(3)}")

    image_height = opt.image_height
    image_width  = opt.image_width
    max_boundary_dist = opt.max_boundary_dist

    src_id_str = str(src_id).zfill(6)
    tgt_id_str = str(tgt_id).zfill(6)

    src_color_image_path         = os.path.join(example_dir, "color",                   src_id_str + ".jpg")
    src_depth_image_path         = os.path.join(example_dir, "depth",                   src_id_str + ".png")
    tgt_color_image_path         = os.path.join(example_dir, "color",                   tgt_id_str + ".jpg")
    tgt_depth_image_path         = os.path.join(example_dir, "depth",                   tgt_id_str + ".png")
    graph_nodes_path             = os.path.join(example_dir, "graph_nodes",             srt_tgt_str + ".bin")
    graph_edges_path             = os.path.join(example_dir, "graph_edges",             srt_tgt_str + ".bin")
    graph_edges_weights_path     = os.path.join(example_dir, "graph_edges_weights",     srt_tgt_str + ".bin")
    graph_clusters_path          = os.path.join(example_dir, "graph_clusters",          srt_tgt_str + ".bin")
    pixel_anchors_path           = os.path.join(example_dir, "pixel_anchors",           srt_tgt_str + ".bin")
    pixel_weights_path           = os.path.join(example_dir, "pixel_weights",           srt_tgt_str + ".bin")

    # Source color and depth
    source, _, cropper = dataset.DeformDataset.load_image(
        src_color_image_path, src_depth_image_path, intrinsics, image_height, image_width
    )

    # Target color and depth (and boundary mask)
    target, target_boundary_mask, _ = dataset.DeformDataset.load_image(
        tgt_color_image_path, tgt_depth_image_path, intrinsics, image_height, image_width, cropper=cropper,
        max_boundary_dist=max_boundary_dist, compute_boundary_mask=True
    )

    # Graph
    graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = dataset.DeformDataset.load_graph_data(
        graph_nodes_path, graph_edges_path, graph_edges_weights_path, None, 
        graph_clusters_path, pixel_anchors_path, pixel_weights_path, cropper
    )

    num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)

    # Update intrinsics to reflect the crops
    fx, fy, cx, cy = image_proc.modify_intrinsics_due_to_cropping(
        intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'], 
        image_height, image_width, original_h=cropper.h, original_w=cropper.w
    )

    intrinsics = np.zeros((4), dtype=np.float32)
    intrinsics[0] = fx
    intrinsics[1] = fy
    intrinsics[2] = cx
    intrinsics[3] = cy

    #####################################################################################################
    # Predict deformation
    #####################################################################################################

    # Move to device and unsqueeze in the batch dimension (to have batch size 1)
    source_cuda               = torch.from_numpy(source).cuda().unsqueeze(0)
    target_cuda               = torch.from_numpy(target).cuda().unsqueeze(0)
    target_boundary_mask_cuda = torch.from_numpy(target_boundary_mask).cuda().unsqueeze(0)
    graph_nodes_cuda          = torch.from_numpy(graph_nodes).cuda().unsqueeze(0)
    graph_edges_cuda          = torch.from_numpy(graph_edges).cuda().unsqueeze(0)
    graph_edges_weights_cuda  = torch.from_numpy(graph_edges_weights).cuda().unsqueeze(0)
    graph_clusters_cuda       = torch.from_numpy(graph_clusters).cuda().unsqueeze(0)
    pixel_anchors_cuda        = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
    pixel_weights_cuda        = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
    intrinsics_cuda           = torch.from_numpy(intrinsics).cuda().unsqueeze(0)

    num_nodes_cuda            = torch.from_numpy(num_nodes).cuda().unsqueeze(0)

    with torch.no_grad():
        model_data = model(
            source_cuda, target_cuda, 
            graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda, 
            pixel_anchors_cuda, pixel_weights_cuda, 
            num_nodes_cuda, intrinsics_cuda, 
            evaluate=True, split="test"
        )

    # Get some of the results
    rotations_pred    = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
    translations_pred = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()
    
    mask_pred = model_data["mask_pred"]
    assert mask_pred is not None, "Make sure use_mask=True in options.py"
    mask_pred = mask_pred.view(-1, opt.image_height, opt.image_width).cpu().numpy()

    # Compute mask gt for mask baseline
    _, source_points, valid_source_points, target_matches, \
        valid_target_matches, valid_correspondences, _, \
            _ = model_data["correspondence_info"]

    target_matches        = target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
    valid_source_points   = valid_source_points.view(-1, opt.image_height, opt.image_width).cpu().numpy()
    valid_target_matches  = valid_target_matches.view(-1, opt.image_height, opt.image_width).cpu().numpy()
    valid_correspondences = valid_correspondences.view(-1, opt.image_height, opt.image_width).cpu().numpy()

    deformed_graph_nodes = graph_nodes + translations_pred

    # Delete tensors to free up memory
    del source_cuda
    del target_cuda
    del target_boundary_mask_cuda
    del graph_nodes_cuda
    del graph_edges_cuda
    del graph_edges_weights_cuda
    del graph_clusters_cuda
    del pixel_anchors_cuda
    del pixel_weights_cuda
    del intrinsics_cuda

    del model

    #####################################################################################################
    # Prepare data
    #####################################################################################################

    #####################################################################################################
    # Source
    #####################################################################################################
    source_flat = np.moveaxis(source, 0, -1).reshape(-1, 6)
    source_points = viz_utils.transform_pointcloud_to_opengl_coords(source_flat[..., 3:])
    source_colors = source_flat[..., :3]

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    source_pcd.colors = o3d.utility.Vector3dVector(source_colors)

    # keep only object using the mask
    valid_source_mask = np.moveaxis(valid_source_points, 0, -1).reshape(-1).astype(np.bool)
    valid_source_points = source_points[valid_source_mask, :]
    valid_source_colors = source_colors[valid_source_mask, :]
    # source object PointCloud
    source_object_pcd = o3d.geometry.PointCloud()
    source_object_pcd.points = o3d.utility.Vector3dVector(valid_source_points)
    source_object_pcd.colors = o3d.utility.Vector3dVector(valid_source_colors)

    # o3d.visualization.draw_geometries([source_pcd])
    # o3d.visualization.draw_geometries([source_object_pcd])

    #####################################################################################################
    # Source warped
    #####################################################################################################
    warped_deform_pred_3d_np = image_proc.warp_deform_3d(
        source, pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred
    )

    source_warped = np.copy(source)
    source_warped[3:, :, :] = warped_deform_pred_3d_np

    # (source) warped RGB-D image
    source_warped = np.moveaxis(source_warped, 0, -1).reshape(-1, 6)
    warped_points = viz_utils.transform_pointcloud_to_opengl_coords(source_warped[..., 3:])
    warped_colors = source_warped[..., :3]
    # Filter points at (0, 0, 0)
    warped_points = warped_points[valid_source_mask]
    warped_colors = warped_colors[valid_source_mask]
    # warped PointCloud
    warped_pcd = o3d.geometry.PointCloud()
    warped_pcd.points = o3d.utility.Vector3dVector(warped_points)
    warped_pcd.paint_uniform_color([1, 0.706, 0]) # warped_pcd.colors = o3d.utility.Vector3dVector(warped_colors)

    # o3d.visualization.draw_geometries([source_object_pcd, warped_pcd])

    ####################################
    # TARGET #
    ####################################
    # target RGB-D image
    target_flat = np.moveaxis(target, 0, -1).reshape(-1, 6)
    target_points = viz_utils.transform_pointcloud_to_opengl_coords(target_flat[..., 3:])
    target_colors = target_flat[..., :3]
    # target PointCloud
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors)

    # o3d.visualization.draw_geometries([target_pcd])

    ####################################
    # GRAPH #
    ####################################
    
    # Transform to OpenGL coords
    graph_nodes = viz_utils.transform_pointcloud_to_opengl_coords(graph_nodes)
    deformed_graph_nodes = viz_utils.transform_pointcloud_to_opengl_coords(deformed_graph_nodes)

    # Graph nodes
    rendered_graph_nodes = []
    for node in graph_nodes:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        mesh_sphere.translate(node)
        rendered_graph_nodes.append(mesh_sphere)
    
    # Merge all different sphere meshes
    rendered_graph_nodes = viz_utils.merge_meshes(rendered_graph_nodes)

    # Graph edges
    edges_pairs = []
    for node_id, edges in enumerate(graph_edges):
        for neighbor_id in edges:
            if neighbor_id == -1:
                break
            edges_pairs.append([node_id, neighbor_id])    

    colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
    line_mesh = line_mesh_utils.LineMesh(graph_nodes, edges_pairs, colors, radius=0.003)
    line_mesh_geoms = line_mesh.cylinder_segments

    # Merge all different line meshes
    line_mesh_geoms = viz_utils.merge_meshes(line_mesh_geoms)

    # o3d.visualization.draw_geometries([rendered_graph_nodes, line_mesh_geoms, source_object_pcd])

    # Combined nodes & edges
    rendered_graph = [rendered_graph_nodes, line_mesh_geoms]

    ####################################
    # Mask
    ####################################
    mask_pred_flat = mask_pred.reshape(-1)
    valid_correspondences = valid_correspondences.reshape(-1).astype(np.bool)

    ####################################
    # Correspondences
    ####################################
    # target matches
    target_matches = np.moveaxis(target_matches, 0, -1).reshape(-1, 3)
    target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)

    ################################
    # "Good" matches
    ################################
    good_mask = valid_correspondences & (mask_pred_flat >= weight_thr)
    good_source_points_corresp  = source_points[good_mask]
    good_target_matches_corresp = target_matches[good_mask]
    good_mask_pred              = mask_pred_flat[good_mask]

    # number of good matches
    n_good_matches = good_source_points_corresp.shape[0]
    # Subsample
    subsample = True
    if subsample:
        N = 2000
        sampled_idxs = np.random.permutation(n_good_matches)[:N]
        good_source_points_corresp  = good_source_points_corresp[sampled_idxs]
        good_target_matches_corresp = good_target_matches_corresp[sampled_idxs]
        good_mask_pred              = good_mask_pred[sampled_idxs]
        n_good_matches = N
    # both good_source and good_target points together into one vector
    good_matches_points = np.concatenate([good_source_points_corresp, good_target_matches_corresp], axis=0)
    good_matches_lines = [[i, i + n_good_matches] for i in range(0, n_good_matches, 1)]

    # --> Create good (unweighted) lines 
    good_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(good_matches_lines))]
    good_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(good_matches_points),
        lines=o3d.utility.Vector2iVector(good_matches_lines),
    )
    good_matches_set.colors = o3d.utility.Vector3dVector(good_matches_colors)

    # --> Create good weighted lines 
    # first, we need to get the proper color coding
    high_color, low_color = np.array([0.0, 0.8, 0]), np.array([0.8, 0, 0.0])

    good_weighted_matches_colors = np.ones_like(good_source_points_corresp)

    weights_normalized = np.maximum(np.minimum(0.5 + (good_mask_pred - weight_thr) / weight_scale, 1.0), 0.0)
    weights_normalized_opposite = 1 - weights_normalized

    good_weighted_matches_colors[:, 0] = weights_normalized * high_color[0] + weights_normalized_opposite * low_color[0]
    good_weighted_matches_colors[:, 1] = weights_normalized * high_color[1] + weights_normalized_opposite * low_color[1]
    good_weighted_matches_colors[:, 2] = weights_normalized * high_color[2] + weights_normalized_opposite * low_color[2]

    good_weighted_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(good_matches_points),
        lines=o3d.utility.Vector2iVector(good_matches_lines),
    )
    good_weighted_matches_set.colors = o3d.utility.Vector3dVector(good_weighted_matches_colors)


    ################################
    # "Bad" matches
    ################################
    bad_mask = valid_correspondences & (mask_pred_flat < weight_thr)
    bad_source_points_corresp  = source_points[bad_mask]
    bad_target_matches_corresp = target_matches[bad_mask]
    bad_mask_pred              = mask_pred_flat[bad_mask]

    # number of good matches
    n_bad_matches = bad_source_points_corresp.shape[0]

    # both good_source and good_target points together into one vector
    bad_matches_points = np.concatenate([bad_source_points_corresp, bad_target_matches_corresp], axis=0)
    bad_matches_lines = [[i, i + n_bad_matches] for i in range(0, n_bad_matches, 1)]

    # --> Create bad (unweighted) lines 
    bad_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(bad_matches_lines))]
    bad_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bad_matches_points),
        lines=o3d.utility.Vector2iVector(bad_matches_lines),
    )
    bad_matches_set.colors = o3d.utility.Vector3dVector(bad_matches_colors)

    # --> Create bad weighted lines 
    # first, we need to get the proper color coding
    high_color, low_color = np.array([0.0, 0.8, 0]), np.array([0.8, 0, 0.0])

    bad_weighted_matches_colors = np.ones_like(bad_source_points_corresp)

    weights_normalized = np.maximum(np.minimum(0.5 + (bad_mask_pred - weight_thr) / weight_scale, 1.0), 0.0)
    weights_normalized_opposite = 1 - weights_normalized

    bad_weighted_matches_colors[:, 0] = weights_normalized * high_color[0] + weights_normalized_opposite * low_color[0]
    bad_weighted_matches_colors[:, 1] = weights_normalized * high_color[1] + weights_normalized_opposite * low_color[1]
    bad_weighted_matches_colors[:, 2] = weights_normalized * high_color[2] + weights_normalized_opposite * low_color[2]

    bad_weighted_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bad_matches_points),
        lines=o3d.utility.Vector2iVector(bad_matches_lines),
    )
    bad_weighted_matches_set.colors = o3d.utility.Vector3dVector(bad_weighted_matches_colors)

    ####################################
    # Generate info for aligning source to target (by interpolating between source and warped source)
    ####################################
    assert warped_points.shape[0] == valid_source_points.shape[0]
    line_segments = warped_points - valid_source_points
    line_segments_unit, line_lengths = line_mesh_utils.normalized(line_segments)
    line_lengths = line_lengths[:, np.newaxis]
    line_lengths = np.repeat(line_lengths, 3, axis=1)

    ####################################
    # Draw 
    ####################################
    geometry_dict = {
        "source_pcd": source_pcd, 
        "source_obj": source_object_pcd, 
        "target_pcd": target_pcd, 
        "graph":      rendered_graph
        # "deformed_graph":    rendered_deformed_graph
    }

    alignment_dict = {
        "valid_source_points": valid_source_points,
        "line_segments_unit":  line_segments_unit,
        "line_lengths":        line_lengths
    }

    matches_dict = {
        "good_matches_set":          good_matches_set,
        "good_weighted_matches_set": good_weighted_matches_set,
        "bad_matches_set":           bad_matches_set,
        "bad_weighted_matches_set":  bad_weighted_matches_set
    }

    #####################################################################################################
    # Open viewer
    #####################################################################################################
    manager = viz_utils.CustomDrawGeometryWithKeyCallback(
        geometry_dict, alignment_dict, matches_dict
    )
    manager.custom_draw_geometry_with_key_callback()
    

if __name__ == "__main__":
    main()