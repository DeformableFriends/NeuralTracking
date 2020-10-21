import sys,os
import json
from torch.utils.data import Dataset
import torch
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import math
from utils import image_proc
from timeit import default_timer as timer
import random
import scipy
import torchvision.transforms.functional as TF
from utils.utils import load_flow, load_graph_nodes, load_graph_edges, load_graph_edges_weights, load_graph_node_deformations, \
                        load_graph_clusters, load_int_image, load_float_image
from utils import image_proc
from NeuralNRT._C import compute_pixel_anchors_geodesic as compute_pixel_anchors_geodesic_c
from NeuralNRT._C import compute_pixel_anchors_euclidean as compute_pixel_anchors_euclidean_c
from NeuralNRT._C import compute_mesh_from_depth as compute_mesh_from_depth_c
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import compute_edges_euclidean as compute_edges_euclidean_c

from utils import utils


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        if len(img.shape) == 2:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2]
        else:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


class DeformDataset(Dataset):
    def __init__(
        self, 
        dataset_base_dir, data_version,
        input_width, input_height, max_boundary_dist
    ):
        self.dataset_base_dir = dataset_base_dir
        self.data_version_json = os.path.join(self.dataset_base_dir, data_version + ".json")
        
        self.input_width = input_width
        self.input_height = input_height

        self.max_boundary_dist = max_boundary_dist

        self.cropper = None
        
        self._load()

    def _load(self):
        with open(self.data_version_json) as f:
            self.labels = json.loads(f.read())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.labels[index]

        src_color_image_path         = os.path.join(self.dataset_base_dir, data["source_color"])
        src_depth_image_path         = os.path.join(self.dataset_base_dir, data["source_depth"])
        tgt_color_image_path         = os.path.join(self.dataset_base_dir, data["target_color"])
        tgt_depth_image_path         = os.path.join(self.dataset_base_dir, data["target_depth"])
        graph_nodes_path             = os.path.join(self.dataset_base_dir, data["graph_nodes"])
        graph_edges_path             = os.path.join(self.dataset_base_dir, data["graph_edges"])
        graph_edges_weights_path     = os.path.join(self.dataset_base_dir, data["graph_edges_weights"])
        graph_node_deformations_path = os.path.join(self.dataset_base_dir, data["graph_node_deformations"])
        graph_clusters_path          = os.path.join(self.dataset_base_dir, data["graph_clusters"])
        pixel_anchors_path           = os.path.join(self.dataset_base_dir, data["pixel_anchors"])
        pixel_weights_path           = os.path.join(self.dataset_base_dir, data["pixel_weights"])
        optical_flow_image_path      = os.path.join(self.dataset_base_dir, data["optical_flow"])
        scene_flow_image_path        = os.path.join(self.dataset_base_dir, data["scene_flow"])

        # Load source, target image and flow.
        source, _, cropper = DeformDataset.load_image(
            src_color_image_path, src_depth_image_path, data["intrinsics"], self.input_height, self.input_width
        )
        target, target_boundary_mask, _ = DeformDataset.load_image(
            tgt_color_image_path, tgt_depth_image_path, data["intrinsics"], self.input_height, self.input_width, cropper=cropper,
            max_boundary_dist=self.max_boundary_dist, compute_boundary_mask=True
        )
        
        optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask = DeformDataset.load_flow(
            optical_flow_image_path, scene_flow_image_path, cropper
        )

        # Load/compute graph.
        graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters, pixel_anchors, pixel_weights = DeformDataset.load_graph_data(
            graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path, 
            graph_clusters_path, pixel_anchors_path, pixel_weights_path, cropper
        )

        # Compute groundtruth transformation for graph nodes.
        num_nodes = graph_nodes.shape[0]

        # Check that flow mask is valid for at least one pixel.
        assert np.sum(optical_flow_mask) > 0, "Zero flow mask for sample: " + json.dumps(data)

        # Store intrinsics.
        fx = data["intrinsics"]["fx"]
        fy = data["intrinsics"]["fy"]
        cx = data["intrinsics"]["cx"]
        cy = data["intrinsics"]["cy"]

        fx, fy, cx, cy = image_proc.modify_intrinsics_due_to_cropping(
            fx, fy, cx, cy, self.input_height, self.input_width, original_h=480, original_w=640
        )

        intrinsics = np.zeros((4), dtype=np.float32)
        intrinsics[0] = fx
        intrinsics[1] = fy
        intrinsics[2] = cx
        intrinsics[3] = cy

        return {
            "source": source, 
            "target": target, 
            "target_boundary_mask": target_boundary_mask,
            "optical_flow_gt": optical_flow_gt, 
            "optical_flow_mask": optical_flow_mask,
            "scene_flow_gt": scene_flow_gt,
            "scene_flow_mask": scene_flow_mask,
            "graph_nodes": graph_nodes, 
            "graph_edges": graph_edges,
            "graph_edges_weights": graph_edges_weights, 
            "graph_node_deformations": graph_node_deformations,
            "graph_clusters": graph_clusters, 
            "pixel_anchors": pixel_anchors, 
            "pixel_weights": pixel_weights, 
            "num_nodes": np.array(num_nodes, dtype=np.int64),
            "intrinsics": intrinsics, 
            "index": np.array(index, dtype=np.int32)
        }

    def get_metadata(self, index):
        return self.labels[index]

    @staticmethod
    def load_image(
        color_image_path, depth_image_path, 
        intrinsics, input_height, input_width, cropper=None,
        max_boundary_dist=0.1, compute_boundary_mask=False
    ):
        # Load images.
        color_image = io.imread(color_image_path) # (h, w, 3)
        depth_image = io.imread(depth_image_path) # (h, w)

        # Backproject depth image.
        depth_image = image_proc.backproject_depth(depth_image, intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]) # (3, h, w)
        depth_image = depth_image.astype(np.float32)
        depth_image = np.moveaxis(depth_image, 0, -1) # (h, w, 3)

        image_size = color_image.shape[:2]

        # Crop, since we need it to be divisible by 64
        if cropper is None:
            cropper = StaticCenterCrop(image_size, (input_height, input_width)) 

        color_image = cropper(color_image)                                                       
        depth_image = cropper(depth_image)

        # Construct the final image.
        image = np.zeros((6, input_height, input_width), dtype=np.float32)
        
        image[:3, :, :] = np.moveaxis(color_image, -1, 0) / 255.0       # (3, h, w)
        assert np.max(image[:3, :, :]) <= 1.0, np.max(image[:3, :, :])
        image[3:, :, :] = np.moveaxis(depth_image, -1, 0)               # (3, h, w)

        if not compute_boundary_mask:
            return image, None, cropper
        else:
            assert max_boundary_dist
            boundary_mask = image_proc.compute_boundary_mask(depth_image, max_boundary_dist)
            return image, boundary_mask, cropper

    @staticmethod
    def load_flow(optical_flow_image_path, scene_flow_image_path, cropper):
        # Load flow images.
        optical_flow_image = load_flow(optical_flow_image_path) # (2, h, w)
        scene_flow_image   = load_flow(scene_flow_image_path)   # (3, h, w)

        # Temporarily move axis for cropping
        optical_flow_image = np.moveaxis(optical_flow_image, 0, -1) # (h, w, 2)
        scene_flow_image   = np.moveaxis(scene_flow_image, 0, -1)   # (h, w, 3)

        # Crop for dimensions to be divisible by 64
        optical_flow_image = cropper(optical_flow_image)
        scene_flow_image   = cropper(scene_flow_image)
        
        # Compute flow mask.
        optical_flow_mask = np.isfinite(optical_flow_image)                                      # (h, w, 2)
        optical_flow_mask = np.logical_and(optical_flow_mask[..., 0], optical_flow_mask[..., 1]) # (h, w)
        optical_flow_mask = optical_flow_mask[..., np.newaxis]                                   # (h, w, 1)
        optical_flow_mask = np.repeat(optical_flow_mask, 2, axis=2)                              # (h, w, 2)

        scene_flow_mask = np.isfinite(scene_flow_image)                                                             # (h, w, 3)
        scene_flow_mask = np.logical_and(scene_flow_mask[..., 0], scene_flow_mask[..., 1], scene_flow_mask[..., 2]) # (h, w)
        scene_flow_mask = scene_flow_mask[..., np.newaxis]                                                          # (h, w, 1) 
        scene_flow_mask = np.repeat(scene_flow_mask, 3, axis=2)                                                     # (h, w, 3)

        # set invalid pixels to zero in the flow image
        optical_flow_image[optical_flow_mask == False] = 0.0
        scene_flow_image[scene_flow_mask == False] = 0.0

        # put channels back in first axis
        optical_flow_image = np.moveaxis(optical_flow_image, -1, 0).astype(np.float32) # (2, h, w)
        optical_flow_mask  = np.moveaxis(optical_flow_mask, -1, 0).astype(np.int64)    # (2, h, w)

        scene_flow_image = np.moveaxis(scene_flow_image, -1, 0).astype(np.float32) # (3, h, w)
        scene_flow_mask  = np.moveaxis(scene_flow_mask, -1, 0).astype(np.int64)    # (3, h, w)

        return optical_flow_image, optical_flow_mask, scene_flow_image, scene_flow_mask

    @staticmethod
    def load_graph_data(
        graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path, graph_clusters_path, 
        pixel_anchors_path, pixel_weights_path, cropper
    ):
        # Load data.
        graph_nodes             = load_graph_nodes(graph_nodes_path)
        graph_edges             = load_graph_edges(graph_edges_path)
        graph_edges_weights     = load_graph_edges_weights(graph_edges_weights_path)
        graph_node_deformations = load_graph_node_deformations(graph_node_deformations_path) if graph_node_deformations_path is not None else None
        graph_clusters          = load_graph_clusters(graph_clusters_path)
        pixel_anchors           = cropper(load_int_image(pixel_anchors_path))
        pixel_weights           = cropper(load_float_image(pixel_weights_path))

        assert np.isfinite(graph_edges_weights).all(), graph_edges_weights
        assert np.isfinite(pixel_weights).all(),       pixel_weights    

        if graph_node_deformations is not None:
            assert np.isfinite(graph_node_deformations).all(), graph_node_deformations
            assert graph_node_deformations.shape[1] == 3
            assert graph_node_deformations.dtype == np.float32

        return graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters, pixel_anchors, pixel_weights

    @staticmethod
    def collate_with_padding(batch):
        batch_size = len(batch)

        # Compute max number of nodes.
        item_keys = 0
        max_num_nodes = 0
        for sample_idx in range(batch_size):
            item_keys = batch[sample_idx].keys()
            num_nodes = batch[sample_idx]["num_nodes"]
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        # Convert merged parts into torch tensors.
        # We pad graph nodes, edges and deformation ground truth with zeros.
        batch_converted = {}

        for key in item_keys:
            if key == "graph_nodes" or key == "graph_edges" or \
                key == "graph_edges_weights" or key == "graph_node_deformations" or \
                    key == "graph_clusters":

                batched_sample = torch.zeros((batch_size, max_num_nodes, batch[0][key].shape[1]), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx, :batch[sample_idx][key].shape[0], :] = torch.from_numpy(batch[sample_idx][key])
                batch_converted[key] = batched_sample

            else:
                batched_sample = torch.zeros((batch_size, *batch[0][key].shape), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx] = torch.from_numpy(batch[sample_idx][key])
                batch_converted[key] = batched_sample

        return [
            batch_converted["source"],
            batch_converted["target"],
            batch_converted["target_boundary_mask"],
            batch_converted["optical_flow_gt"],
            batch_converted["optical_flow_mask"],
            batch_converted["scene_flow_gt"],
            batch_converted["scene_flow_mask"],
            batch_converted["graph_nodes"],
            batch_converted["graph_edges"],
            batch_converted["graph_edges_weights"],
            batch_converted["graph_node_deformations"],
            batch_converted["graph_clusters"],
            batch_converted["pixel_anchors"],
            batch_converted["pixel_weights"],
            batch_converted["num_nodes"],
            batch_converted["intrinsics"],
            batch_converted["index"]
        ]