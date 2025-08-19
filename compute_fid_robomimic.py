import argparse
import h5py
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import re

def stream_h5_images_from_demos(h5_path, dataset_path="obs/agentview_image"):
    """Yield batches of images from a single HDF5 file containing multiple demos."""
    with h5py.File(h5_path, "r") as f:
        # Sort demo keys numerically
        demo_keys = sorted(
            [k for k in f['data'].keys() if re.match(r"demo_\d+", k)],
            key=lambda x: int(x.split("_")[1])
        )
        for demo_key in demo_keys:
            # if dataset_path not in f[demo_key]:
            #     raise KeyError(
            #         f"Dataset '{dataset_path}' not found in {demo_key}. "
            #         f"Available: {list(f[demo_key].keys())}"
            #     )
            data = f['data'][demo_key][dataset_path][:]
            yield data  # shape: (T, H, W, C)

def normalize_and_to_tensor(images):
    """Convert images to uint8 and torch tensor NCHW format."""
    if images.dtype != np.uint8:
        images = (images - images.min()) / (images.max() - images.min()) * 255
        images = images.astype(np.uint8)
    # Convert NHWC -> NCHW
    tensor = torch.tensor(images).permute(0, 3, 1, 2)
    return tensor

def compute_fid(real_file, fake_file, dataset_path="obs/agentview_image", batch_size=256):
    fid = FrechetInceptionDistance(feature=2048).to("cuda" if torch.cuda.is_available() else "cpu")
    device = fid.device

    # Process real
    with tqdm(desc="Processing real demos") as pbar:
        for imgs in stream_h5_images_from_demos(real_file, dataset_path):
            imgs_tensor = normalize_and_to_tensor(imgs).to(device)
            for i in range(0, len(imgs_tensor), batch_size):
                fid.update(imgs_tensor[i:i+batch_size], real=True)
            pbar.update()

    # Process fake
    with tqdm(desc="Processing fake demos") as pbar:
        for imgs in stream_h5_images_from_demos(fake_file, dataset_path):
            imgs_tensor = normalize_and_to_tensor(imgs).to(device)
            for i in range(0, len(imgs_tensor), batch_size):
                fid.update(imgs_tensor[i:i+batch_size], real=False)
            pbar.update()

    return fid.compute().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score between two single H5 demo files.")
    parser.add_argument("--real", type=str, required=True, help="H5 file containing real demos")
    parser.add_argument("--fake", type=str, required=True, help="H5 file containing fake demos")
    parser.add_argument("--dataset_path", type=str, default="obs/agentview_image",
                        help="Path inside each demo group to images (default: obs/agentview_image)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for FID computation")
    args = parser.parse_args()

    fid_score = compute_fid(
        real_file=args.real,
        fake_file=args.fake,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size
    )

    print(f"FID score: {fid_score:.4f}")
