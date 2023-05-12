# Setting
```bash
git clone https://github.com/SakodaShintaro/nerfacc --recursive
cd nerfacc

apt install sudo -y
sudo apt update
sudo apt install -y vim python3-pip wget unzip ninja-build
pip3 install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html

pip3 install imageio tqdm lpips
pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

bash scripts/download_example_data.sh

export PYTHONPATH=.
python3 examples/train_ngp_nerf_occ.py --scene lego --data_root data/nerf_synthetic
```
