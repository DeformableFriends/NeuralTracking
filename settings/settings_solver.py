#####################################################################################################################
# GENERAL OPTIONS
#####################################################################################################################
# Do validation?
do_validation = True

# Freeze parts of the model
freeze_optical_flow_net   = False
freeze_mask_net           = False

# Shuffle batch
shuffle = True

# If true, we'll not run Gauss-Newton optimization
skip_solver = False


#####################################################################################################################
# BASELINE OPTIONS
#####################################################################################################################
min_neg_flowed_source_to_target_dist = 0.3
max_pos_flowed_source_to_target_dist = 0.1
max_boundary_dist = 0.10


#####################################################################################################################
# SOLVER OPTIONS
#####################################################################################################################
gn_depth_sampling_mode = "bilinear" # "bilinear" or "nearest"
gn_max_depth = 6.0
gn_min_nodes = 4
gn_max_nodes = 300
gn_max_matches_train = 10000 
gn_max_matches_train_per_batch = 100000 
gn_max_matches_eval = 10000 
gn_max_warped_points = 100000
gn_debug = False
gn_print_timings = False

gn_use_edge_weighting = False
gn_check_condition_num = False
gn_break_on_condition_num = True
gn_max_condition_num = 1e6

gn_remove_clusters_with_few_matches = True
gn_min_num_correspondences_per_cluster = 2000

gn_invalidate_too_far_away_translations = True
gn_max_mean_translation_error = 0.5


#####################################################################################################################
# Losses
#####################################################################################################################
# Architecture parameters
use_flow_loss      = True; lambda_flow = 5.0
use_graph_loss     = True; lambda_graph = 2.0
use_warp_loss      = True; lambda_warp  = 2.0
use_mask           = False
use_mask_loss      = False; lambda_mask = 1000.0 # one of the baselines in the paper

flow_loss_type = 'RobustL1'

# Only applies when use_mask_loss=True (i.e., one of the baselines)
mask_neg_wrt_pos_weight = 0.05 # None if you don't wanna use a fixed weight, but to let it depend on the #negs vs # pos

# Keep only those matches for which the mask prediction is above a threshold (Only applies if evaluating)
threshold_mask_predictions           = False;  threshold = 0.30
patchwise_threshold_mask_predictions = False; patch_size = 8
assert not (threshold_mask_predictions and patchwise_threshold_mask_predictions)


#####################################################################################################################
# Learning parameters
#####################################################################################################################
use_adam = False
use_batch_norm = False
batch_size = 4
evaluation_frequency = 2000 # in number of iterations
epochs = 15
learning_rate = 1e-5
use_lr_scheduler = True
step_lr = 10000
weight_decay = 0.0
momentum = 0.9
margin = 1.0