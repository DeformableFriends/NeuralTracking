import os

#####################################################################################################################
# DATA OPTIONS
#####################################################################################################################
dataset_base_dir    = "/cluster_HDD/lothlann/nonrigid/public/"
workspace           = "."
experiments_dir     = os.path.join(workspace, "experiments")

image_width = 640
image_height = 448
num_worker_threads = 6
num_threads = 4

num_samples_eval = 700 

#####################################################################################################################
# MODEL INFO
#####################################################################################################################

# Info for a saved model
# - In train.py, this info is only used if use_pretrained_model=True
# - In generate.py, evaluate.py or example_viz.py, it is used regardless of the value of use_pretrained_model

use_pretrained_model = True # used only in train.py

model_module_to_load = "full_model" # A: "only_flow_net", B: "full_model"
model_name           = "model_A"    # your model's name, e.g., "model_A", "chairs_things"
model_iteration      = 0            # iteration number of the model you want to load

saved_model = os.path.join(experiments_dir, "models", model_name, f"{model_name}_{model_iteration}.pt")

#####################################################################################################################
# TRAINING OPTIONS
#####################################################################################################################
mode = "0_flow" # ["0_flow", "1_solver", "2_mask", "3_refine"]

if mode == "0_flow":
    from settings.settings_flow import *
elif mode == "1_solver":
    from settings.settings_solver import *
elif mode == "2_mask":
    from settings.settings_mask import *
elif mode == "3_refine":
    from settings.settings_refine import *
elif mode == "4_your_custom_settings":
    # from settings.4_your_custom_settings import *
    pass

#####################################################################################################################
# Print options
#####################################################################################################################

# GPU id
def print_hyperparams():
    print("HYPERPARAMETERS:")
    print()

    print("\tnum_worker_threads           ", num_worker_threads)

    if use_pretrained_model:
        print("\tPretrained model              \"{}\"".format(saved_model))
        print("\tModel part to load:          ", model_module_to_load)
        print("\tfreeze_optical_flow_net      ", freeze_optical_flow_net)
        print("\tfreeze_mask_net              ", freeze_mask_net)
    else:
        print("\tPretrained model              None")
    
    print()
    print("\tuse_adam                     ", use_adam)
    print("\tbatch_size                   ", batch_size)
    print("\tevaluation_frequency         ", evaluation_frequency)
    print("\tepochs                       ", epochs)
    print("\tlearning_rate                ", learning_rate)
    if use_lr_scheduler:
        print("\tstep_lr                      ", step_lr)
    else:
        print("\tstep_lr                      ", "None")
    print("\tweight_decay                 ", weight_decay)
    print()
    print("\tgn_max_matches_train         ", gn_max_matches_train)
    print("\tgn_max_matches_eval          ", gn_max_matches_eval)
    print("\tgn_depth_sampling_mode       ", gn_depth_sampling_mode)
    print("\tgn_use_edge_weighting        ", gn_use_edge_weighting)
    print("\tgn_remove_clusters           ", gn_remove_clusters_with_few_matches)
    print()
    print("\tmin_neg_flowed_dist          ", min_neg_flowed_source_to_target_dist)
    print("\tmax_neg_flowed_dist          ", max_pos_flowed_source_to_target_dist)
    print("\tmax_boundary_dist            ", max_boundary_dist)
    print()
    print("\tflow_loss_type               ", flow_loss_type)
    print("\tuse_flow_loss                ", use_flow_loss, "\t", lambda_flow)
    print("\tuse_graph_loss               ", use_graph_loss, "\t", lambda_graph)
    print("\tuse_warp_loss                ", use_warp_loss, "\t", lambda_warp)
    print("\tuse_mask_loss                ", use_mask_loss, "\t", lambda_mask)
    print()
    print("\tuse_mask                     ", use_mask)