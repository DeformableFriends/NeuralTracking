import sys,os
import argparse
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from shutil import copyfile
import random
import sys
import numpy as np
from timeit import default_timer as timer
from skimage import io
import math
import torchvision.models as models

from model import evaluate
from model import dataset
from utils import utils
import options as opt
from utils.snapshot_manager import SnapshotManager
from utils.time_statistics import TimeStatistics
from utils import nnutils
from model.model import DeformNet
from model.loss import DeformLoss
import utils.query as query


if __name__ == "__main__":
    torch.set_num_threads(opt.num_threads)
    torch.backends.cudnn.benchmark = False

    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', action='store', dest='train_dir', help='Provide a subfolder with training data')
    parser.add_argument('--val_dir', action='store', dest='val_dir', help='Provide a subfolder with SPARSE validation data')
    parser.add_argument('--experiment', action='store', dest='experiment', help='Provide an experiment name')
    parser.add_argument('--date', action='store', dest='date', help='Provide a date in the format %Y-%m-%d (if you do not want current date)')

    args = parser.parse_args()
    
    # Train set 
    train_dir = args.train_dir

    # Val set
    val_dir = args.val_dir

    experiment_name = args.experiment
    if not experiment_name:
        clock = datetime.now().strftime('%H-%M-%S')
        experiment_name = "{}_default".format(clock)

    date = args.date
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')

    #####################################################################################
    # Ask user input regarding the use of data augmentation
    #####################################################################################
    # Confirm hyperparameters
    opt.print_hyperparams()

    print()
    print("train_dir        ", train_dir)
    print("val_dir          ", val_dir)
    print()

    # use_current_hyper = query.query_yes_no("\nThe above hyperparameters will be used. Do you wish to continue?", "yes")
    # if not use_current_hyper:
    #     print("Exiting. Please modify options.py and run this script again.")
    #     exit()

    #####################################################################################
    # Creating tf writer and folders 
    #####################################################################################
    # Writer initialization.
    tf_runs = os.path.join(opt.experiments_dir, "tf_runs")
    log_name = "{0}_{1}".format(date, experiment_name)
    log_dir = os.path.join(tf_runs, log_name)

    train_log_dir = log_dir + "/" + train_dir
    val_log_dir = log_dir + "/" + val_dir
    if train_log_dir == val_log_dir:
        train_log_dir = train_log_dir + "_0"
        val_log_dir = val_log_dir + "_1"
    
    train_writer = SummaryWriter(train_log_dir)
    val_writer = SummaryWriter(val_log_dir)

    # Copy the current options to the log directory.
    options_file_in = os.path.abspath(os.path.join(os.path.dirname(__file__), "options.py"))
    options_file_out = os.path.join(log_dir, "options.py")
    copyfile(options_file_in, options_file_out)

    # Creation of model dir.
    training_models = os.path.join(opt.experiments_dir, "models")
    if not os.path.exists(training_models): os.mkdir(training_models)
    saving_model_dir = os.path.join(training_models, log_name)
    if not os.path.exists(saving_model_dir): os.mkdir(saving_model_dir)    

    #####################################################################################
    # Initializing: model, criterion, optimizer, learning scheduler...
    #####################################################################################
    # Load model, loss and optimizer.
    saved_model = opt.saved_model

    iteration_number = 0

    model = DeformNet().cuda()

    if opt.use_pretrained_model:
        assert os.path.isfile(saved_model), "\nModel {} does not exist. Please train a model from scratch or specify a valid path to a model.".format(saved_model)
        pretrained_dict = torch.load(saved_model)
        
        if "network-chairs-things.pytorch" in saved_model:
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

    # Criterion.
    criterion = DeformLoss(opt.lambda_flow, opt.lambda_graph, opt.lambda_warp, opt.lambda_mask, opt.flow_loss_type)

    # Count parameters.
    n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_model_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))
    
    n_all_flownet_params = int(sum([np.prod(p.size()) for p in model.flow_net.parameters()]))
    n_trainable_flownet_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.flow_net.parameters())]))
    print("-> Flow network: {0} / {1}".format(n_trainable_flownet_params, n_all_flownet_params))

    n_all_masknet_params = int(sum([np.prod(p.size()) for p in model.mask_net.parameters()]))
    n_trainable_masknet_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.mask_net.parameters())]))
    print("-> Mask network: {0} / {1}".format(n_trainable_masknet_params, n_all_masknet_params))
    print()

    # Set up optimizer.
    if opt.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # Initialize training.
    train_writer.add_text("/hyperparams",
                    "Batch size: " + str(opt.batch_size)
                    + ",\nLearning rate:" + str(opt.learning_rate)
                    + ",\nEpochs: " + str(opt.epochs)
                    + ",\nuse_flow_loss: " + str(opt.use_flow_loss)
                    + ",\nuse_graph_loss: " + str(opt.use_graph_loss)
                    + ",\nuse_mask: " + str(opt.use_mask)
                    + ",\nuse_mask_loss: " + str(opt.use_mask_loss))

    # Initialize snaphost manager for model snapshot creation.
    snapshot_manager = SnapshotManager(log_name, saving_model_dir)
    
    # We count the execution time between evaluations.
    time_statistics = TimeStatistics()

    # Learning rate scheduler.
    if opt.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_lr, gamma=0.1, last_epoch=-1)
        for i in range(iteration_number):
            scheduler.step()

    # Compute the number of worker threads for data loading.
    # 0 means that the base thread does all the job (that makes sense when hdf5 is already loaded into memory).
    num_train_workers = opt.num_worker_threads
    num_val_workers = opt.num_worker_threads

    #####################################################################################
    # Create datasets and dataloaders
    #####################################################################################
    complete_cycle_start = timer()

    #####################################################################################
    # VAL dataset
    #####################################################################################
    val_dataset = dataset.DeformDataset(
        opt.dataset_base_dir, val_dir, 
        opt.image_width, opt.image_height, opt.max_boundary_dist
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, shuffle=opt.shuffle, 
        batch_size=opt.batch_size, num_workers=num_val_workers, 
        collate_fn=dataset.DeformDataset.collate_with_padding, pin_memory=True
    )

    print("Num. validation samples: {0}".format(len(val_dataset)))

    if len(val_dataset) < opt.batch_size:
        print()
        print("Reduce the batch_size, since we only have {} validation samples but you indicated a batch_size of {}".format(
            len(val_dataset), opt.batch_size)
        )
        exit()

    #####################################################################################
    # TRAIN dataset
    #####################################################################################
    train_dataset = dataset.DeformDataset(
        opt.dataset_base_dir, train_dir, 
        opt.image_width, opt.image_height, opt.max_boundary_dist
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=opt.batch_size, 
        shuffle=opt.shuffle, num_workers=num_train_workers, 
        collate_fn=dataset.DeformDataset.collate_with_padding, pin_memory=True
    )

    print("Num. training samples: {0}".format(len(train_dataset)))
    print()

    if len(train_dataset) < opt.batch_size:
        print()
        print("Reduce the batch_size, since we only have {} training samples but you indicated a batch_size of {}".format(
            len(train_dataset), opt.batch_size)
        )
        exit()

    # Execute training.
    try:
        for epoch in range(0, opt.epochs):
            print()
            print()
            print("Epoch: {0}".format(epoch))

            num_consecutive_all_invalid_batches = 0
            
            model.train()
            for i, data in enumerate(train_dataloader):
                #####################################################################################
                # Validation.
                #####################################################################################
                if opt.do_validation and iteration_number % opt.evaluation_frequency == 0:
                    model.eval()

                    eval_start = timer()

                    # Compute train and validation metrics.
                    num_samples = opt.num_samples_eval
                    num_eval_batches = math.ceil(num_samples / opt.batch_size) # We evaluate on approximately 1000 samples.

                    print()
                    print("Train evaluation")
                    train_losses, train_metrics = evaluate.evaluate(model, criterion, train_dataloader, num_eval_batches, "train")
                    
                    print()
                    print("Val   evaluation")
                    val_losses, val_metrics     = evaluate.evaluate(model, criterion, val_dataloader, num_eval_batches, "val")

                    train_writer.add_scalar('Loss/Loss',        train_losses["total"],  iteration_number)
                    train_writer.add_scalar('Loss/Flow',        train_losses["flow"],   iteration_number)
                    train_writer.add_scalar('Loss/Graph',       train_losses["graph"], iteration_number)
                    train_writer.add_scalar('Loss/Warp',        train_losses["warp"],   iteration_number)
                    train_writer.add_scalar('Loss/Mask',        train_losses["mask"],   iteration_number)
                    
                    train_writer.add_scalar('Metrics/EPE_2D_0',             train_metrics["epe2d_0"],      iteration_number)
                    train_writer.add_scalar('Metrics/EPE_2D_2',             train_metrics["epe2d_2"],      iteration_number)
                    train_writer.add_scalar('Metrics/Graph_Error_3D',       train_metrics["epe3d"],        iteration_number)
                    train_writer.add_scalar('Metrics/EPE_3D',               train_metrics["epe_warp"],     iteration_number)
                    train_writer.add_scalar('Metrics/ValidRatio',           train_metrics["valid_ratio"],  iteration_number)

                    val_writer.add_scalar('Loss/Loss',      val_losses["total"],    iteration_number)
                    val_writer.add_scalar('Loss/Flow',      val_losses["flow"],     iteration_number)
                    val_writer.add_scalar('Loss/Graph',     val_losses["graph"],   iteration_number)
                    val_writer.add_scalar('Loss/Warp',      val_losses["warp"],     iteration_number)
                    val_writer.add_scalar('Loss/Mask',      val_losses["mask"],     iteration_number)
                    
                    val_writer.add_scalar('Metrics/EPE_2D_0',             val_metrics["epe2d_0"],      iteration_number)
                    val_writer.add_scalar('Metrics/EPE_2D_2',             val_metrics["epe2d_2"],      iteration_number)
                    val_writer.add_scalar('Metrics/Graph_Error_3D',       val_metrics["epe3d"],        iteration_number)
                    val_writer.add_scalar('Metrics/EPE_3D',               val_metrics["epe_warp"],     iteration_number)
                    val_writer.add_scalar('Metrics/ValidRatio',           val_metrics["valid_ratio"],  iteration_number)
                
                    print()
                    print()
                    print("Epoch number {0}, Iteration number {1}".format(epoch, iteration_number))
                    print("{:<40} {}".format("Current Train Loss TOTAL",    train_losses["total"]))
                    print("{:<40} {}".format("Current Train Loss FLOW",     train_losses["flow"]))
                    print("{:<40} {}".format("Current Train Loss GRAPH",    train_losses["graph"]))
                    print("{:<40} {}".format("Current Train Loss WARP",     train_losses["warp"]))
                    print("{:<40} {}".format("Current Train Loss MASK",     train_losses["mask"]))
                    print()
                    print("{:<40} {}".format("Current Train EPE 2D_0",            train_metrics["epe2d_0"]))
                    print("{:<40} {}".format("Current Train EPE 2D_2",            train_metrics["epe2d_2"]))
                    print("{:<40} {}".format("Current Train EPE 3D",              train_metrics["epe3d"]))
                    print("{:<40} {}".format("Current Train EPE Warp",            train_metrics["epe_warp"]))
                    print("{:<40} {}".format("Current Train Solver Success Rate", train_metrics["valid_ratio"]))
                    print()

                    print("{:<40} {}".format("Current Val Loss TOTAL",      val_losses["total"]))
                    print("{:<40} {}".format("Current Val Loss FLOW",       val_losses["flow"]))
                    print("{:<40} {}".format("Current Val Loss GRAPH",      val_losses["graph"]))
                    print("{:<40} {}".format("Current Val Loss WARP",       val_losses["warp"]))
                    print("{:<40} {}".format("Current Val Loss MASK",       val_losses["mask"]))
                    print()
                    print("{:<40} {}".format("Current Val EPE 2D_0",            val_metrics["epe2d_0"]))
                    print("{:<40} {}".format("Current Val EPE 2D_2",            val_metrics["epe2d_2"]))
                    print("{:<40} {}".format("Current Val EPE 3D",              val_metrics["epe3d"]))
                    print("{:<40} {}".format("Current Val EPE Warp",            val_metrics["epe_warp"]))
                    print("{:<40} {}".format("Current Val Solver Success Rate", val_metrics["valid_ratio"]))

                    print()

                    time_statistics.eval_duration = timer() - eval_start

                    # We compute the time of IO as the complete time, subtracted by all processing time.
                    time_statistics.io_duration += (timer() - complete_cycle_start - time_statistics.train_duration - time_statistics.eval_duration)
                    
                    # Set CUDA_LAUNCH_BLOCKING=1 environmental variable for reliable timings. 
                    print("Cycle duration (s): {0:3f} (IO: {1:3f}, TRAIN: {2:3f}, EVAL: {3:3f})".format(
                        timer() - time_statistics.start_time, time_statistics.io_duration, time_statistics.train_duration, time_statistics.eval_duration
                    ))
                    print("FORWARD: {0:3f}, LOSS: {1:3f}, BACKWARD: {2:3f}".format(
                        time_statistics.forward_duration, time_statistics.loss_eval_duration, time_statistics.backward_duration
                    ))                       
                    print()   

                    time_statistics = TimeStatistics()
                    complete_cycle_start = timer()

                    sys.stdout.flush()

                    model.train()
                
                else:
                    sys.stdout.write("\r############# Train iteration: {0} / {1} (of Epoch {2}) - {3}".format(
                        iteration_number % opt.evaluation_frequency + 1, opt.evaluation_frequency, epoch, experiment_name)
                    )
                    sys.stdout.flush()

                #####################################################################################
                # Train.
                #####################################################################################
                # Data loading.
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

                train_batch_start = timer()

                #####################################################################################
                # Forward pass.
                #####################################################################################
                train_batch_forward_pass = timer()

                model_data = model(
                    source, target, 
                    graph_nodes, graph_edges, graph_edges_weights, graph_clusters, 
                    pixel_anchors, pixel_weights, 
                    num_nodes, intrinsics
                )
                
                time_statistics.forward_duration += (timer() - train_batch_forward_pass)

                # Invalidate too for too far away estimations, since they can produce
                # noisy gradient information.
                if opt.gn_invalidate_too_far_away_translations:
                    with torch.no_grad():
                        batch_size = model_data["node_translations"].shape[0]
                        for i in range(batch_size):
                            if not model_data["valid_solve"][i]: continue

                            num_nodes_i = int(num_nodes[i])
                            assert num_nodes_i > 0

                            diff = model_data["node_translations"][i, :num_nodes_i, :] - translations_gt[i, :num_nodes_i, :]
                            epe = torch.norm(diff, p=2, dim=1)
                            mean_error = epe.sum().item() / num_nodes_i

                            if mean_error > opt.gn_max_mean_translation_error:
                                print("\t\tToo big mean translation error: {}".format(mean_error))
                                model_data["valid_solve"][i] = 0
                
                with torch.no_grad():
                    # Downscale groundtruth flow
                    flow_gts, flow_masks = nnutils.downscale_gt_flow(
                        optical_flow_gt, optical_flow_mask, opt.image_height, opt.image_width
                    )    

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
                                    
                #####################################################################################
                # Loss.
                #####################################################################################
                train_batch_loss_eval = timer()

                # Compute Loss
                loss = criterion(
                    flow_gts, model_data["flow_data"], flow_masks,
                    translations_gt, model_data["node_translations"], model_data["deformations_validity"],
                    deformed_points_gt, model_data["deformed_points_pred"], deformed_points_mask,
                    model_data["valid_solve"], num_nodes, 
                    model_data["mask_pred"], mask_gt, valid_mask_pixels
                )

                time_statistics.loss_eval_duration += (timer() - train_batch_loss_eval)

                #####################################################################################
                # Backprop.
                #####################################################################################
                train_batch_backprop = timer()

                # We only backprop if any of the losses is non-zero.
                if opt.use_flow_loss or opt.use_mask_loss or torch.sum(model_data["valid_solve"]) > 0:
                    optimizer.zero_grad()
                    loss.backward()
                 
                    optimizer.step()
                    if opt.use_lr_scheduler: scheduler.step()
                    
                else:
                    print("No valid loss, skipping backpropagation!")

                time_statistics.backward_duration += (timer() - train_batch_backprop)

                time_statistics.train_duration += (timer() - train_batch_start)

                if iteration_number % opt.evaluation_frequency == 0:
                    # Store the latest model snapshot, if the required elased time has passed.
                    snapshot_manager.save_model(model, iteration_number)  

                iteration_number = iteration_number + 1

            print()
            print("Epoch {} complete".format(epoch))
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")
        
    except (KeyboardInterrupt, TypeError, ConnectionResetError) as err:
        # We also save the latest model snapshot at interruption.
        snapshot_manager.save_model(model, iteration_number, final_iteration=True)  
        raise err

    train_writer.close()
    val_writer.close()

    print()
    print("I'm done")