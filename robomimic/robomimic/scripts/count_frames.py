import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_frame_count(file):
    with h5py.File(file) as f:
        frame_count = 0
        breakpoint()
        for i in range(0, 200):
            frame_count += f[f'data/demo_{i}/actions'].shape[0]

    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract images/segmentations from hdf5 files into flat folders")
    parser.add_argument("--root", type=str, required=True, help="root with all hdf5 files to extract")
    args = parser.parse_args()

    hdf5_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".hdf5")]
    hdf5_files.sort()
    if not hdf5_files:
        print(f"No hdf5 files found in {args.root}")
        return

    total = 0
    for file in hdf5_files:
        count = get_frame_count(file)
        print(f"{file}: {count}")
        total += count

    print(f"Total frame count: {total}")


if __name__ == "__main__":
    main()
