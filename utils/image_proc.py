import sys, os
import struct
import numpy as np
import math

from NeuralNRT._C import backproject_depth_ushort as backproject_depth_ushort_c
from NeuralNRT._C import backproject_depth_float as backproject_depth_float_c
from NeuralNRT._C import warp_flow as warp_flow_c
from NeuralNRT._C import warp_rigid as warp_rigid_c
from NeuralNRT._C import warp_3d as warp_3d_c
from utils import utils 


def warp_flow_py(image, flow, mask):
    # We assume:
    #               image shape (3, 480, 640)
    #               flow shape  (2, 480, 640)
    #               mask shape  (2, 480, 640)

    image_copy = np.copy(image)
    flow_copy = np.copy(flow)
    mask_copy = np.copy(mask)

    # print(type(image_copy), image_copy.dtype)

    image_copy = np.moveaxis(image_copy, 0, -1) # (h, w, 3)
    flow_copy = np.moveaxis(flow_copy, 0, -1)
    mask_copy = np.moveaxis(mask_copy, 0, -1)

    h, w = flow_copy.shape[0], flow_copy.shape[1]

    assert image_copy.shape[0] == flow_copy.shape[0] and image_copy.shape[1] == flow_copy.shape[1]
    assert mask_copy.shape[0] == flow_copy.shape[0] and mask_copy.shape[1] == flow_copy.shape[1]

    image_warped = np.zeros(image_copy.shape, dtype=np.float32)
    weights_warped = np.zeros((h, w, 1), dtype=np.float32)
    
    for v in range(h):
        for u in range(w):

            if mask_copy is not None and (not mask_copy[v, u, 0] or not mask_copy[v, u, 1]):
                continue

            u_warped = u + flow_copy[v, u, 0]
            v_warped = v + flow_copy[v, u, 1]

            u0 = math.floor(u_warped)
            u1 = u0 + 1
            v0 = math.floor(v_warped)
            v1 = v0 + 1

            if not in_bounds((u0, v0), h, w) or not in_bounds((u0, v1), h, w) \
                or not in_bounds((u1, v0), h, w) or not in_bounds((u1, v1), h, w):
                continue 

            du = u_warped - u0
            dv = v_warped - v0

            w00 = (1 - du) * (1 - dv)
            w01 = (1 - du) * dv
            w10 = du * (1 - dv)
            w11 = du * dv

            c = image_copy[v, u]
            image_warped[v0, u0] += w00 * c
            image_warped[v0, u1] += w10 * c
            image_warped[v1, u0] += w01 * c
            image_warped[v1, u1] += w11 * c
            weights_warped[v0, u0] += w00
            weights_warped[v0, u1] += w10
            weights_warped[v1, u0] += w01
            weights_warped[v1, u1] += w11

    zero_weights = weights_warped == 0
    weights_warped[zero_weights] = 1.0
    image_warped /= weights_warped
    image_warped[np.repeat(zero_weights, 3, axis=2)] = 1.0

    return np.moveaxis(image_warped, -1, 0) # (3, h, w)


def warp_flow(image, flow, mask):
    # We assume:
    #               image shape (3, h, w)
    #               flow shape  (2, h, w)
    #               mask shape  (2, h, w)

    assert image.shape[1] == flow.shape[1] and image.shape[2] == flow.shape[2]
    assert mask.shape[1] == flow.shape[1] and mask.shape[2] == flow.shape[2]

    image_warped = warp_flow_c(image, flow, mask)

    return image_warped


def warp_deform_py(image, pixel_anchors, pixel_weights, node_positions, node_rotations, node_translations, fx, fy, cx, cy):
    # We assume:
    #               image               shape (6, h, w)
    #               pixel_anchors       shape (h, w, 4)
    #               pixel_weights       shape (h, w, 4)
    #               node_positions      shape (num_nodes, 3)
    #               node_rotations      shape (num_nodes, 3, 3)
    #               node_translations   shape (num_nodes, 3)

    image_copy = np.copy(image)

    h = image.shape[1]
    w = image.shape[2]
    num_pixels = h * w

    fx, fy, cx, cy = modify_intrinsics_due_to_cropping(fx, fy, cx, cy, h, w, original_h=480, original_w=640)

    invalid_pixel_anchors = pixel_anchors < 0
    filtered_pixel_anchors = np.copy(pixel_anchors)
    filtered_pixel_anchors[invalid_pixel_anchors] = 0 

    # Warp the image pixels using graph poses.
    image_points = image[3:, :, :]
    image_points = np.moveaxis(image_points, 0, -1).reshape(num_pixels, 3, 1)
    deformed_points = np.zeros((num_pixels, 3, 1), dtype=image.dtype) 
    
    num_nodes = node_translations.shape[0]
    node_translations = node_translations.reshape(num_nodes, 3, 1)

    for k in range(4):
        node_idxs_k = pixel_anchors.reshape(num_pixels, 4)[:, k]
        nodes_k = node_positions[node_idxs_k].reshape(num_pixels, 3, 1)
        
        # Compute deformed point contribution.                    
        rotated_points_k = np.matmul(node_rotations[node_idxs_k], image_points - nodes_k) # (num_pixels, 3, 1)
        deformed_points_k = rotated_points_k + nodes_k + node_translations[node_idxs_k]
        interpolation_weights = pixel_weights.reshape(num_pixels, 4)[:, k].reshape(num_pixels, 1, 1)
        deformed_points += np.repeat(interpolation_weights, 3, axis=1) * deformed_points_k # (num_pixels, 3, 1)

    deformed_points = deformed_points.reshape(h, w, 3)
    deformed_points = np.moveaxis(deformed_points, -1, 0)

    # Compute the warped image.
    image_warped = np.zeros((h, w, 3), dtype=np.float32)
    weights_warped = np.zeros((h, w, 1), dtype=np.float32)

    for v in range(h):
        for u in range(w):
            p = image[3:, v, u].reshape(3, 1)
            if p[2] <= 0.0: continue

            p_def = deformed_points[:, v, u]
            if p_def[2] <= 0.0: continue

            if invalid_pixel_anchors[v, u, 0] and invalid_pixel_anchors[v, u, 1] and invalid_pixel_anchors[v, u, 2] and invalid_pixel_anchors[v, u, 3]:
                continue

            u_warped = fx * p_def[0] / p_def[2] + cx
            v_warped = fy * p_def[1] / p_def[2] + cy

            u0 = math.floor(u_warped)
            u1 = u0 + 1
            v0 = math.floor(v_warped)
            v1 = v0 + 1

            if not in_bounds((u0, v0), h, w) or not in_bounds((u0, v1), h, w) \
                or not in_bounds((u1, v0), h, w) or not in_bounds((u1, v1), h, w):
                continue 

            du = u_warped - u0
            dv = v_warped - v0

            w00 = (1 - du)*(1 - dv)
            w01 = (1 - du)*dv
            w10 = du*(1 - dv)
            w11 = du*dv

            c = image_copy[:3, v, u]

            image_warped[v0, u0] += w00 * c
            image_warped[v0, u1] += w10 * c
            image_warped[v1, u0] += w01 * c
            image_warped[v1, u1] += w11 * c
            weights_warped[v0, u0] += w00
            weights_warped[v0, u1] += w10
            weights_warped[v1, u0] += w01
            weights_warped[v1, u1] += w11

    zero_weights = weights_warped == 0
    weights_warped[zero_weights] = 1.0
    image_warped /= weights_warped
    image_warped[np.repeat(zero_weights, 3, axis=2)] = 1.0
    
    return np.moveaxis(image_warped, -1, 0)


def warp_deform(image, pixel_anchors, pixel_weights, node_positions, node_rotations, node_translations, fx, fy, cx, cy):
    # We assume:
    #               image               shape (6, h, w)
    #               pixel_anchors       shape (h, w, 4)
    #               pixel_weights       shape (h, w, 4)
    #               node_positions      shape (num_nodes, 3)
    #               node_rotations      shape (num_nodes, 3, 3)
    #               node_translations   shape (num_nodes, 3)

    image_copy = np.copy(image)

    h = image_copy.shape[1]
    w = image_copy.shape[2]
    num_pixels = h * w

    fx, fy, cx, cy = modify_intrinsics_due_to_cropping(fx, fy, cx, cy, h, w, original_h=480, original_w=640)

    # invalid_pixel_anchors = pixel_anchors < 0
    # filtered_pixel_anchors = np.copy(pixel_anchors)
    # filtered_pixel_anchors[invalid_pixel_anchors] = 0 

    # Warp the image pixels using graph poses.
    image_points = image_copy[3:, :, :]
    image_points = np.moveaxis(image_points, 0, -1).reshape(num_pixels, 3, 1)
    deformed_points = np.zeros((num_pixels, 3, 1), dtype=image_copy.dtype) 
    
    num_nodes = node_translations.shape[0]
    node_translations = node_translations.reshape(num_nodes, 3, 1)

    for k in range(4):
        node_idxs_k = pixel_anchors.reshape(num_pixels, 4)[:, k]
        nodes_k = node_positions[node_idxs_k].reshape(num_pixels, 3, 1)
        
        # Compute deformed point contribution.                    
        rotated_points_k = np.matmul(node_rotations[node_idxs_k], image_points - nodes_k) # (num_pixels, 3, 1)
        deformed_points_k = rotated_points_k + nodes_k + node_translations[node_idxs_k]
        interpolation_weights = pixel_weights.reshape(num_pixels, 4)[:, k].reshape(num_pixels, 1, 1)
        deformed_points += np.repeat(interpolation_weights, 3, axis=1) * deformed_points_k # (num_pixels, 3, 1)

    deformed_points = deformed_points.reshape(h, w, 3)
    deformed_points = np.moveaxis(deformed_points, -1, 0)

    # point_validity = np.logical_not(np.all(invalid_pixel_anchors, axis=2))
    point_validity = ~np.any(pixel_anchors == -1, axis=2)

    # Compute the warped image.
    image_warped = warp_3d_c(image_copy, deformed_points, point_validity, fx, fy, cx, cy)
    
    return image_warped


def warp_deform_3d(image, pixel_anchors, pixel_weights, node_positions, node_rotations, node_translations):
    # We assume:
    #               image               shape (6, h, w)
    #               pixel_anchors       shape (h, w, 4)
    #               pixel_weights       shape (h, w, 4)
    #               node_positions      shape (num_nodes, 3)
    #               node_rotations      shape (num_nodes, 3, 3)
    #               node_translations   shape (num_nodes, 3)

    assert image.shape[0] == 6 and len(image.shape) == 3, image.shape
    assert pixel_anchors.shape[-1] == 4 and len(pixel_anchors.shape) == 3
    assert pixel_weights.shape[-1] == 4 and len(pixel_weights.shape) == 3
    assert node_positions.shape[-1] == 3
    assert node_rotations.shape[1] == 3 and node_rotations.shape[2] == 3 
    assert node_translations.shape[-1] == 3

    image_copy = np.copy(image)

    h = image_copy.shape[1]
    w = image_copy.shape[2]
    num_pixels = h * w

    # invalid_pixel_anchors = pixel_anchors < 0
    # filtered_pixel_anchors = np.copy(pixel_anchors)
    # filtered_pixel_anchors[invalid_pixel_anchors] = 0 

    # filtered_pixel_anchors[np.any(pixel_anchors == -1, axis=2)]

    # Warp the image pixels using graph poses.
    image_points = image_copy[3:, :, :]
    image_points = np.moveaxis(image_points, 0, -1).reshape(num_pixels, 3, 1)
    deformed_points = np.zeros((num_pixels, 3, 1), dtype=image_copy.dtype) 
    
    num_nodes = node_translations.shape[0]
    node_translations = node_translations.reshape(num_nodes, 3, 1)

    for k in range(4):
        node_idxs_k = pixel_anchors.reshape(num_pixels, 4)[:, k]
        nodes_k = node_positions[node_idxs_k].reshape(num_pixels, 3, 1)

        # Compute deformed point contribution.                    
        rotated_points_k = np.matmul(node_rotations[node_idxs_k], image_points - nodes_k) # (num_pixels, 3, 1)
        deformed_points_k = rotated_points_k + nodes_k + node_translations[node_idxs_k]
        interpolation_weights = pixel_weights.reshape(num_pixels, 4)[:, k].reshape(num_pixels, 1, 1)
        deformed_points += np.repeat(interpolation_weights, 3, axis=1) * deformed_points_k # (num_pixels, 3, 1)

    deformed_points = deformed_points.reshape(h, w, 3)
    deformed_points = np.moveaxis(deformed_points, -1, 0)

    point_validity = np.all(pixel_anchors != -1, axis=2)
    point_validity = np.repeat(point_validity.reshape(1, h, w), 3, axis=0)

    deformed_points[~point_validity] = 0.0

    return deformed_points


def modify_intrinsics_due_to_cropping(fx, fy, cx, cy, h, w, original_h=480, original_w=640):
    # Modify intrinsics
    delta_height = (h - original_h) / 2
    cy += delta_height

    delta_width = (w / original_w) / 2
    cx += delta_width

    return fx, fy, cx, cy


def backproject_depth_py(depth_image, fx, fy, cx, cy, normalizer = 1000.0):
    assert len(depth_image.shape) == 2
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    point_image = np.zeros((3, height, width))
    for y in range(height):
        for x in range(width):
            depth = depth_image[y, x] / normalizer
            if depth > 0:
                pos_x = depth * (x - cx) / fx
                pos_y = depth * (y - cy) / fy
                pos_z = depth

                point_image[0, y, x] = pos_x
                point_image[1, y, x] = pos_y
                point_image[2, y, x] = pos_z

    return point_image


def backproject_depth(depth_image, fx, fy, cx, cy, normalizer = 1000.0):
    assert len(depth_image.shape) == 2
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    point_image = np.zeros((3, height, width), dtype=np.float32)

    if depth_image.dtype == np.float32:
        backproject_depth_float_c(depth_image, point_image, fx, fy, cx, cy)
    else:
        backproject_depth_ushort_c(depth_image, point_image, fx, fy, cx, cy, normalizer)

    return point_image


def compute_boundary_mask(depth_image, max_distance):
    depth_image_copy = np.copy(depth_image)

    boundary_mask = np.zeros(depth_image_copy.shape, dtype=bool)

    shift_right = np.zeros_like(depth_image_copy)
    shift_left  = np.zeros_like(depth_image_copy)
    shift_down  = np.zeros_like(depth_image_copy)
    shift_up    = np.zeros_like(depth_image_copy)

    shift_right[:,1:,:] = depth_image_copy[:,:-1,:]
    shift_left[:,:-1,:] = depth_image_copy[:,1:,:]
    shift_down[1:,:,:]  = depth_image_copy[:-1,:,:]
    shift_up[:-1,:,:]   = depth_image_copy[1:,:,:]

    horizontal_dist = np.linalg.norm(shift_left - shift_right, axis=2)
    vertical_dist   = np.linalg.norm(shift_up - shift_down, axis=2)

    assert np.isfinite(horizontal_dist).all()
    assert np.isfinite(vertical_dist).all()

    boundary_horizontal_mask = horizontal_dist > max_distance
    boundary_vertical_mask   = vertical_dist   > max_distance

    boundary_mask = boundary_horizontal_mask | boundary_vertical_mask


    return boundary_mask[None,...]