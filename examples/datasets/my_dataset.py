"""
Copyright (c) 2023 Shintaro Sakoda.
"""

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

from .utils import Rays


def _load_renderings(root_fp: str):
    """Load images from disk."""
    assert root_fp.startswith("/"), "False : root_fp.startswith('/')"

    data_dir = root_fp
    pose_npy = np.load(f"{data_dir}/cams_meta.npy")

    images = []
    cam2worlds = []
    Ks = []

    for i in range(len(pose_npy)):
        fname = f"{data_dir}/dense/images/{i:08d}.png"
        rgba = imageio.imread(fname)
        images.append(rgba)

        curr_data = pose_npy[i][0:12].reshape((3, 4))
        add_row = np.array([0, 0, 0, 1])
        cam2world = np.vstack((curr_data, add_row))
        cam2worlds.append(cam2world)
        Ks.append(pose_npy[i][12:21].reshape((3, 3)))

    images = np.stack(images, axis=0)
    cam2worlds = np.stack(cam2worlds, axis=0)
    Ks = np.stack(Ks, axis=0)

    return images, cam2worlds, Ks


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    OPENGL_CAMERA = True

    def __init__(
        self,
        root_fp: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        resize_factor: int = 1,
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.training = (num_rays is not None)
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.Ks = _load_renderings(root_fp)
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.Ks = torch.from_numpy(self.Ks).to(torch.float32)

        assert resize_factor >= 1
        if resize_factor != 1:
            resize = Resize((self.images.shape[1] // resize_factor,
                            self.images.shape[2] // resize_factor))
            self.images = resize(self.images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            self.Ks /= resize_factor
        self.HEIGHT, self.WIDTH = self.images.shape[1:3]

        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.Ks = self.Ks.to(device)
        self.K = self.Ks[0]

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgb, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = rgb
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgb = torch.reshape(rgb, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,    # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
