# render.py (DIFIX Modified with Height Selection)
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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, height_level: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # =========================================================
        # 1. 确定筛选前缀 (根据 H 参数)
        # =========================================================
        target_prefix = "virtual" # 默认为 virtual
        output_folder_name = "virtual_views"

        if height_level == 1: 
            target_prefix = "virtual1_"
            output_folder_name = "virtual_views_h1"
        elif height_level == 2: 
            target_prefix = "virtual2_"
            output_folder_name = "virtual_views_h2"
        elif height_level == 3: 
            target_prefix = "virtual3_"
            output_folder_name = "virtual_views_h3"
        elif height_level == 0: 
            target_prefix = "virtual"
            output_folder_name = "virtual_views"
        
        
        print(f"\n[DIFIX Selection] Target Prefix: '{target_prefix}' (Height Level: {height_level})")

        # =========================================================
        # 2. 从所有相机中筛选目标
        # =========================================================
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        render_targets = []

        for cam in all_cameras:
            # 只要名字以指定前缀开头，就加入渲染列表
            if cam.image_name.startswith(target_prefix):
                render_targets.append(cam)
        
        # 按名称排序，保证渲染顺序
        render_targets.sort(key=lambda x: x.image_name)

        if len(render_targets) == 0:
            print(f"[Warning] No cameras found starting with '{target_prefix}'.")
            return

        print(f"[DIFIX Selection] Found {len(render_targets)} target cameras to render.")

        # =========================================================

        
        # 创建一个目标ID集合，查找速度快
        target_ids = set([id(cam) for cam in render_targets])
        
        deleted_count = 0
        for cam in all_cameras:
            # 如果当前相机不在我们要渲染的目标列表里
            if id(cam) not in target_ids:
                if hasattr(cam, 'original_image'):
                    # 彻底释放显存
                    cam.original_image = None 
                    deleted_count += 1
        
        # 强制清理 PyTorch 缓存
        torch.cuda.empty_cache()
        print(f"[DIFIX Memory] Deleted {deleted_count} unused images from VRAM. Starting render...")

        # =========================================================
        # 4. 执行渲染
        # =========================================================
        render_set(dataset.model_path, output_folder_name, scene.loaded_iter, render_targets, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # 如果需要保存GT用于对比，请取消下面这行的注释
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 1. 执行渲染
        res = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = res["render"]

        # 2. 获取 GT (如果需要)
        # 注意：因为我们在外面只删除了“非目标”的图片，所以当前 view.original_image 是存在的
        # gt = view.original_image[0:3, :, :]

        # 3. 保存渲染结果
        save_name = view.image_name
        if not save_name.endswith(".png"):
            base = os.path.splitext(save_name)[0]
            save_name = base + ".png"
            
        torchvision.utils.save_image(rendering, os.path.join(render_path, save_name))
        
        # 如果需要保存GT:
        # torchvision.utils.save_image(gt, os.path.join(gts_path, save_name))

        # =========================================================
        # [DIFIX 循环内清理] 用完即弃
        # 这张图渲染完并保存后，它的原始数据就没有价值了，立刻删掉防止显存堆积
        # =========================================================
        if hasattr(view, 'original_image'):
            view.original_image = None
        
        # 清理渲染结果张量
        del rendering
        del res
        # del gt
        
        # 每一帧都清理显存缓存，这对 6000x4000 分辨率至关重要
        torch.cuda.empty_cache()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # [修改] 添加 -H 参数
    parser.add_argument("-H", "--height_level", type=int, default=0, help="0: All virtual, 1: virtual1_, 2: virtual2_, 3: virtual3_")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.height_level)

