import argparse
import h5py
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from glob import glob
import os
from tqdm import tqdm

def stream_h5_images(folder, dataset_name="rgb_frames", img_idx=2):
    """Yield images from all H5 files in a folder as numpy arrays."""
    for file in sorted(glob(os.path.join(folder, "*.h5"))):
        with h5py.File(file, "r") as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file}. Available: {list(f.keys())}")

            if f[dataset_name].shape[1] == 1:
                img_idx = 0
            data = f[dataset_name][:, img_idx]
            yield data

def normalize_and_to_tensor(images):
    """Convert images to uint8 and torch tensor NCHW format."""
    if images.dtype != np.uint8:
        images = (images - images.min()) / (images.max() - images.min()) * 255
        images = images.astype(np.uint8)
    # Convert NHWC -> NCHW for PyTorch
    tensor = torch.tensor(images).permute(0, 3, 1, 2)
    return tensor

def compute_fid(real_folder, fake_folder, dataset_name="rgb_frames", batch_size=256, img_idx=2):
    fid = FrechetInceptionDistance(feature=2048).to("cuda" if torch.cuda.is_available() else "cpu")

    device = fid.device

    num_real = len(glob(os.path.join(real_folder, "*.h5")))
    num_fake = len(glob(os.path.join(fake_folder, "*.h5")))

    # Process real images
    with tqdm(total=num_real, desc="Processing real", unit="img") as pbar:
        for imgs in stream_h5_images(real_folder, dataset_name):
            imgs_tensor = normalize_and_to_tensor(imgs).to(device)
            for i in range(0, len(imgs_tensor), batch_size):
                fid.update(imgs_tensor[i:i+batch_size], real=True)
            pbar.update()

    # Process fake images
    with tqdm(total=num_fake, desc="Processing fake", unit="img") as pbar:
        for imgs in stream_h5_images(fake_folder, dataset_name):
            imgs_tensor = normalize_and_to_tensor(imgs).to(device)
            for i in range(0, len(imgs_tensor), batch_size):
                fid.update(imgs_tensor[i:i+batch_size], real=False)
            pbar.update()

    return fid.compute().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score between two folders of H5 files.")
    parser.add_argument("--real_folder", type=str, required=True, help="Folder containing real H5 files")
    parser.add_argument("--fake_folder", type=str, required=True, help="Folder containing generated H5 files")
    parser.add_argument("--dataset_name", type=str, default="rgb_frames", help="Dataset key in the H5 files (default: rgb_frames)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for FID computation")
    parser.add_argument("--img_idx", type=int, default=2, help="Index corresponding to camera to use from dataset")
    args = parser.parse_args()

    fid_score = compute_fid(
        real_folder=args.real_folder,
        fake_folder=args.fake_folder,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size, 
        img_idx=args.img_idx
    )

    print(f"FID score: {fid_score:.4f}")
