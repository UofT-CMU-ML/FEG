ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 1000,
    deformation_lr_init = 0.0004,
    deformation_lr_final = 0.0000004,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    momentum = 0.9,
    iterations = 6000,
    percent_dense = 0.01,
    opacity_reset_interval = 7000,
    position_lr_max_steps = 7000,
    prune_interval = 6000 
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 64,
     'resolution': [64, 64, 64, 100]

    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
)
