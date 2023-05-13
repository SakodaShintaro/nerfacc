import imageio.v2 as imageio
import numpy as np
from glob import glob
import argparse
import os
from tqdm import tqdm
import subprocess


def concat_two_images(save_dir: str) -> None:
    """Concatenate two images into one."""
    images_pred = sorted(glob(f"{save_dir}/pred/*.png"))
    images_gt = sorted(glob(f"{save_dir}/gt/*.png"))
    os.makedirs(f"{save_dir}/concat", exist_ok=True)

    for i in tqdm(range(len(images_pred))):
        image1 = imageio.imread(images_pred[i])
        image2 = imageio.imread(images_gt[i])
        image = np.concatenate([image1, image2], axis=1)
        imageio.imwrite(f"{save_dir}/concat/{i:08d}.png", image)


def make_movie(save_dir: str) -> None:
    """Make a movie from images."""
    command = "ffmpeg -y " \
        "-r 10 " \
        "-f " \
        "image2 -i %08d.png " \
        "-vcodec libx264 " \
        "-crf 25 " \
        "-pix_fmt yuv420p " \
        "-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" " \
        "../output.mp4"
    print(command)
    subprocess.run(command, shell=True, cwd=save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    args = parser.parse_args()
    save_dir = args.save_dir
    concat_two_images(save_dir)
    make_movie(f"{save_dir}/concat")
