"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

from examples.utils import (
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator
import os

from datasets.my_dataset import SubjectLoader

from concat_and_make_movie import concat_two_images, make_movie

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    required=True,
    help="the root dir of the dataset",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./train_result"
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

# training parameters
max_steps = 20000
init_batch_size = 1024
target_sample_batch_size = 1 << 18
weight_decay = 1e-6
# scene parameters
s = 6
aabb = torch.tensor([-s, -s, -s, s, s, s], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3
alpha_thre = 0.0
cone_angle = 0.0

train_dataset = SubjectLoader(
    root_fp=args.data_root,
    num_rays=init_batch_size,
    device=device,
    resize_factor=1
)

test_dataset = SubjectLoader(
    root_fp=args.data_root,
    num_rays=None,
    device=device,
    resize_factor=8
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

os.makedirs(args.save_dir, exist_ok=True)
loss_log = open(f"{args.save_dir}/loss_log.txt", "w")

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 10000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print_str = f"elapsed_time={elapsed_time:.2f}s | step={step} | " \
            f"loss={loss:.5f} | psnr={psnr:.2f} | " \
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | " \
            f"max_depth={depth.max():.3f} | "
        loss_log.write(print_str + "\n")
        loss_log.flush()
        print(print_str)

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        estimator.eval()

        torch.save(radiance_field.state_dict(), f"{args.save_dir}/radiance_field.pt")
        torch.save(estimator.state_dict(), f"{args.save_dir}/estimator.pt")

        psnrs = []
        lpips = []
        with torch.no_grad():
            save_image_dir = f"{args.save_dir}/test_images"
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                os.makedirs(f"{save_image_dir}/pred", exist_ok=True)
                imageio.imwrite(
                    f"{save_image_dir}/pred/{i:08d}.png",
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                os.makedirs(f"{save_image_dir}/gt", exist_ok=True)
                imageio.imwrite(
                    f"{save_image_dir}/gt/{i:08d}.png",
                    (pixels.cpu().numpy() * 255).astype(np.uint8),
                )
                os.makedirs(f"{save_image_dir}/error", exist_ok=True)
                imageio.imwrite(
                    f"{save_image_dir}/error/{i:08d}.png",
                    ((rgb - pixels).norm(dim=-1).cpu().numpy() * 255).astype(np.uint8),
                )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
        concat_two_images(save_image_dir)
        make_movie(f"{save_image_dir}/concat")
