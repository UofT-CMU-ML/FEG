#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef
from models.networks import CNN_decoder
import torch.nn.functional as F

import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def count_gaussian_params(gaussian_model):
    """
    Counts trainable parameters in the custom GaussianModel
    by summing .numel() for all Tensors with requires_grad=True.
    """
    total = 0
    # We iterate through the GaussianModel attributes:
    for name, val in vars(gaussian_model).items():
        if isinstance(val, torch.Tensor) and val.requires_grad:
            total += val.numel()
    return total

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0

    #f3dgs
    train_cameras = scene.getTrainCameras()  # This returns a FourDGSdataset
    random_idx = randint(0, len(train_cameras)-1)
    viewpoint_cam = train_cameras[random_idx]
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]
    print('feature out dim', feature_out_dim) # 256

    feature_in_dim = int(feature_out_dim/2)
    # feature_in_dim = feature_out_dim
    print('feature in dim', feature_in_dim) # 256
    cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim) # in dim and out dim expected to be 256

    cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)

    ### ADDED CODE ###
    # Count parameters of the CNN decoder (an nn.Module)
    decoder_param_count = sum(p.numel() for p in cnn_decoder.parameters() if p.requires_grad)

    # Count parameters in the GaussianModel (not an nn.Module)
    gaussian_param_count = count_gaussian_params(gaussians)

    # Combine them:
    total_params = decoder_param_count + gaussian_param_count

    # Approx. size in MB if float32 => 4 bytes per param
    model_size_mb = (total_params * 4) / (1024**2)

    print(f"\n[INFO] Parameter Count:")
    print(f"  CNN Decoder parameters: {decoder_param_count}")
    print(f"  GaussianModel parameters: {gaussian_param_count}")
    print(f"  TOTAL trainable parameters: {total_params}")
    print(f"  Approx. model size: {model_size_mb:.2f} MB (float32)\n")

    # Log to TensorBoard
    if tb_writer is not None:
        tb_writer.add_scalar("Model/decoder_param_count", decoder_param_count, first_iter)
        tb_writer.add_scalar("Model/gaussian_param_count", gaussian_param_count, first_iter)
        tb_writer.add_scalar("Model/total_param_count", total_params, first_iter)
        tb_writer.add_scalar("Model/model_size_MB_float32", model_size_mb, first_iter)
    ### END ADDED CODE ###

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda:0")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()

    print('scene mode is:', scene.mode)
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            idx = 0
        else:
            idx = randint(0, len(viewpoint_stack)-1)

        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        #f3dgs
        features_list = []
        gt_features_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, opt.no_deformation)
            feature_map, image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            #f3dgs
            gt_feature = viewpoint_cam.semantic_feature.cuda()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            #f3dgs
            features_list.append(feature_map)
            gt_features_list.append(gt_feature)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)
        feature_map = torch.cat(features_list, 0)
        gt_feature_map = torch.cat(gt_features_list, 0)

        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
        feature_map = cnn_decoder(feature_map)
        # fmap shape: torch.Size([256, 51, 64]) 
        # masks shape: torch.Size([1, 1, 512, 640]) 
        feature_masking = True
        
        if feature_masking:
            mask_down = F.interpolate(masks.float(), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='nearest')
            # Now mask_down has shape [1, 1, 51, 64]
            # Remove the batch dimension if necessary and expand along the channel dimenion.
            mask_down = mask_down.squeeze(0)  # shape becomes [1, 51, 64]
            mask_down = mask_down.repeat(gt_feature_map.shape[0], 1, 1)
            Ll1_feature = torch.abs((feature_map * mask_down) - (gt_feature_map * mask_down)).mean()
        else:
            Ll1_feature = torch.abs((feature_map - gt_feature_map)).mean()
        
        Ll1 = l1_loss(rendered_images, gt_images, masks)
        
        if (gt_depths!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        elif scene.mode == 'binocular':
            rendered_depths[rendered_depths!=0] = 1 / rendered_depths[rendered_depths!=0]
            gt_depths[gt_depths!=0] = 1 / gt_depths[gt_depths!=0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)
        elif scene.mode == 'monocular':
            rendered_depths_reshape = rendered_depths.reshape(-1, 1)
            gt_depths_reshape = gt_depths.reshape(-1, 1)
            mask_tmp = mask.reshape(-1)
            rendered_depths_reshape, gt_depths_reshape = rendered_depths_reshape[mask_tmp!=0, :], gt_depths_reshape[mask_tmp!=0, :]
            depth_loss =  0.001 * (1 - pearson_corrcoef(gt_depths_reshape, rendered_depths_reshape))
        else:
            raise ValueError(f"{scene.mode} is not implemented.")
        
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)
        
        if opt.use_tv_and_depth_loss:
            loss = Ll1 + depth_loss + tv_loss
        else:
            loss = Ll1

        psnr_ = psnr(rendered_images, gt_images, masks).mean().double()        
        
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(2e-2, 2e-2, 2e-2)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images,gt_images)
            loss = (1.0 - opt.lambda_dssim) * loss + opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(rendered_images,gt_images,lpips_model)
            loss += opt.lambda_lpips * lpipsloss

        # Fine stage, feature loss is on by default (if not disabled)
        # Coarse stage, feature loss is off by default, unless explicitly enabled
        if not opt.no_feature_loss and (opt.use_feature_loss_for_coarse or stage == "fine"):
            loss = loss + Ll1_feature
        
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "Loss_feature": f"{Ll1_feature:.{7}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print('saving the cnn decoder')
                torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cnn_decoder_optimizer.step()
                cnn_decoder_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if iteration % 30== 0:
            torch.cuda.empty_cache()

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    if not args.no_fine:
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    '''
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda:0"), 0.0, 1.0)
                    mask = viewpoint.mask.to("cuda:0")
                    
                    image, gt_image, mask = image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
        '''

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3000,4000, 5000, 6000, 9000, 10000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
