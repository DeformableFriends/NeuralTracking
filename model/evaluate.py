import torch
import torch.nn as nn
import numpy as np
import math
import os
import sys

import options as opt
from model import dataset
from utils import nnutils


def evaluate(model, criterion, dataloader, batch_num, split):
    dataset_obj = dataloader.dataset
    dataset_batch_size = dataloader.batch_size
    total_size = len(dataset_obj)

    # Losses
    loss_sum = 0.0
    loss_flow_sum = 0.0
    loss_graph_sum = 0.0
    loss_warp_sum = 0.0
    loss_mask_sum = 0.0

    max_num_batches = int(math.ceil(total_size / dataset_batch_size))
    total_num_batches = batch_num if batch_num != -1 else max_num_batches
    total_num_batches = min(max_num_batches, total_num_batches)

    # Metrics
    epe2d_sum_0 = 0.0
    epe2d_sum_2 = 0.0
    epe3d_sum = 0.0
    epe_warp_sum = 0.0

    total_num_pixels_0 = 0
    total_num_pixels_2 = 0
    total_num_nodes = 0
    total_num_points = 0

    num_valid_solves = 0
    num_total_solves = 0

    total_corres_weight_sum = 0.0
    total_corres_valid_num = 0
    
    print()

    for i, data in enumerate(dataloader):
        if i >= total_num_batches: 
            break

        sys.stdout.write("\r############# Eval iteration: {0} / {1}".format(i + 1, total_num_batches))
        sys.stdout.flush()

        source, target, target_boundary_mask, \
            optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask, \
                    graph_nodes, graph_edges, graph_edges_weights, translations_gt, graph_clusters, \
                        pixel_anchors, pixel_weights, num_nodes, intrinsics, sample_idx = data
        
        source               = source.cuda()
        target               = target.cuda()
        target_boundary_mask = target_boundary_mask.cuda()
        optical_flow_gt      = optical_flow_gt.cuda()
        optical_flow_mask    = optical_flow_mask.cuda()
        scene_flow_gt        = scene_flow_gt.cuda()
        scene_flow_mask      = scene_flow_mask.cuda()
        graph_nodes          = graph_nodes.cuda()
        graph_edges          = graph_edges.cuda()
        graph_edges_weights  = graph_edges_weights.cuda()
        translations_gt      = translations_gt.cuda()
        graph_clusters       = graph_clusters.cuda()
        pixel_anchors        = pixel_anchors.cuda()
        pixel_weights        = pixel_weights.cuda()
        intrinsics           = intrinsics.cuda()

        batch_size = source.shape[0]

        # Build data for coarser level
        assert opt.image_height % 64 == 0 and opt.image_width % 64 == 0
        optical_flow_gt2 = torch.nn.functional.interpolate(input=optical_flow_gt.clone() / 20.0, size=(opt.image_height//4, opt.image_width//4), mode='nearest')
        optical_flow_mask2 = torch.nn.functional.interpolate(input=optical_flow_mask.clone().float(), size=(opt.image_height//4, opt.image_width//4), mode='nearest').bool()
        assert(torch.isfinite(optical_flow_gt2).all().item())
        assert(torch.isfinite(optical_flow_mask2).all().item())

        with torch.no_grad():
            # Predictions.
            model_data = model(
                source, target, 
                graph_nodes, graph_edges, graph_edges_weights, graph_clusters, 
                pixel_anchors, pixel_weights, 
                num_nodes, intrinsics, 
                evaluate=True, split=split
            )
            optical_flow_pred2 = model_data["flow_data"][0]
            optical_flow_pred = 20.0 * torch.nn.functional.interpolate(input=optical_flow_pred2, size=(opt.image_height, opt.image_width), mode='bilinear', align_corners=False)
            
            translations_pred = model_data["node_translations"]

            total_corres_weight_sum += model_data["weight_info"]["total_corres_weight"]
            total_corres_valid_num += model_data["weight_info"]["total_corres_num"]

            # Compute mask gt for mask baseline
            xy_coords_warped, source_points, valid_source_points, target_matches, \
                valid_target_matches, valid_correspondences, deformed_points_idxs, \
                    deformed_points_subsampled = model_data["correspondence_info"]

            mask_gt, valid_mask_pixels = nnutils.compute_baseline_mask_gt(
                xy_coords_warped, 
                target_matches, valid_target_matches,
                source_points, valid_source_points,
                scene_flow_gt, scene_flow_mask, target_boundary_mask,
                opt.max_pos_flowed_source_to_target_dist, opt.min_neg_flowed_source_to_target_dist
            )

            # Compute deformed point gt
            deformed_points_gt, deformed_points_mask = nnutils.compute_deformed_points_gt(
                source_points, scene_flow_gt, 
                model_data["valid_solve"], valid_correspondences, 
                deformed_points_idxs, deformed_points_subsampled
            )

            # Loss.
            loss, loss_flow, loss_graph, loss_warp, loss_mask = criterion(
                [optical_flow_gt], [optical_flow_pred], [optical_flow_mask],
                translations_gt, model_data["node_translations"], model_data["deformations_validity"],
                deformed_points_gt, model_data["deformed_points_pred"], deformed_points_mask,
                model_data["valid_solve"], num_nodes, 
                model_data["mask_pred"], mask_gt, valid_mask_pixels,
                evaluate=True
            )

            loss_sum            += loss.item()
            loss_flow_sum       += loss_flow.item()     if opt.use_flow_loss        else -1
            loss_graph_sum      += loss_graph.item()    if opt.use_graph_loss       else -1
            loss_warp_sum       += loss_warp.item()     if opt.use_warp_loss        else -1
            loss_mask_sum       += loss_mask.item()     if opt.use_mask_loss        else -1

            # Metrics.
            # A.1) End Point Error in Optical Flow for FINEST level
            epe2d_dict = criterion.epe_2d(optical_flow_gt, optical_flow_pred, optical_flow_mask) 
            epe2d_sum_0        += epe2d_dict["sum"]
            total_num_pixels_0 += epe2d_dict["num"]

            # A.2) End Point Error in Optical Flow for PYRAMID level 2 (4 times lower rez than finest level)
            epe2d_dict = criterion.epe_2d(optical_flow_gt2, optical_flow_pred2, optical_flow_mask2) 
            epe2d_sum_2        += epe2d_dict["sum"]
            total_num_pixels_2 += epe2d_dict["num"]

            # B) Deformation translation/angle difference.
            # Important: We also evaluate nodes that were filtered at optimization (and were assigned
            # identity deformation). 
            for k in range(batch_size):
                # We validate node deformation of both valid and invalid solves (for invalid
                # solves, the prediction should be identity transformation). 
                num_total_solves += 1

                t_gt   = translations_gt[k].view(1, -1, 3)
                t_pred = translations_pred[k].view(1, -1, 3)
                deformations_validity = model_data["deformations_validity"][k].view(1, -1)

                # For evaluation of all nodes (valid or invalid), we take all nodes into account.
                deformations_validity_all = torch.zeros_like(deformations_validity)
                deformations_validity_all[:, :int(num_nodes[k])] = 1

                epe3d_dict = criterion.epe_3d(t_gt, t_pred, deformations_validity_all)
                epe3d_sum       += epe3d_dict["sum"]
                total_num_nodes += epe3d_dict["num"]
                
                # If valid solve, add to valid solves.
                if not model_data["valid_solve"][k]: continue
                num_valid_solves += 1

            # C) End Point Error in Warped 3D Points
            epe_warp_dict = criterion.epe_warp(deformed_points_gt, model_data["deformed_points_pred"], deformed_points_mask) 
            if epe_warp_dict is not None:
                epe_warp_sum     += epe_warp_dict["sum"]
                total_num_points += epe_warp_dict["num"]
        
    # Losses
    loss_avg            = loss_sum        / total_num_batches
    loss_flow_avg       = loss_flow_sum   / total_num_batches
    loss_graph_avg      = loss_graph_sum  / total_num_batches
    loss_warp_avg       = loss_warp_sum   / total_num_batches
    loss_mask_avg       = loss_mask_sum   / total_num_batches
    
    losses = {
        "total":    loss_avg,
        "flow":     loss_flow_avg,
        "graph":    loss_graph_avg,
        "warp":     loss_warp_avg,
        "mask":     loss_mask_avg
    }

    # Metrics.
    epe2d_avg_0       = epe2d_sum_0 / total_num_pixels_0        if total_num_pixels_0 > 0 else -1.0
    epe2d_avg_2       = epe2d_sum_2 / total_num_pixels_2        if total_num_pixels_2 > 0 else -1.0
    epe3d_avg         = epe3d_sum / total_num_nodes             if total_num_nodes    > 0 else -1.0
    epe_warp_avg      = epe_warp_sum / total_num_points         if total_num_points   > 0 else -1.0
    valid_ratio       = num_valid_solves / num_total_solves     if num_total_solves   > 0 else -1

    if total_corres_valid_num > 0:
        print(" Average correspondence weight: {0:.3f}".format(total_corres_weight_sum / total_corres_valid_num))

    metrics = {
        "epe2d_0": epe2d_avg_0,
        "epe2d_2": epe2d_avg_2,
        "epe3d": epe3d_avg,
        "epe_warp": epe_warp_avg,
        "num_valid_solves": num_valid_solves,
        "num_total_solves": num_total_solves,
        "valid_ratio": valid_ratio,
    }

    return losses, metrics


if __name__ == "__main__":
    pass