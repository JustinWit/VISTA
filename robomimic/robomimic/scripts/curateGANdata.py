import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_test_data(hdf5_path, out_agentview, out_wristview, counters):
    with h5py.File(hdf5_path, "r") as f:
        # Adjust keys if needed
        domainA_agentview_imgs = f["data/domainA/obs/agentview_image"]                # (T, H, W, 3)
        domainA_wrist_imgs = f["data/domainA/obs/robot0_eye_in_hand_image"]           # (T, H, W, 3)
        domainA_agentview_segs = f["data/domainA/obs/agentview_segmentation_final"]   # (T, H, W)
        domainA_wrist_segs = f["data/domainA/obs/robot0_eye_in_hand_segmentation_final"]  # (T, H, W)
        
        domainB_agentview_imgs = f["data/domainB/obs/agentview_image"]                # (T, H, W, 3)
        domainB_wrist_imgs = f["data/domainB/obs/robot0_eye_in_hand_image"]           # (T, H, W, 3)
        domainB_agentview_segs = f["data/domainB/obs/agentview_segmentation_final"]   # (T, H, W)
        domainB_wrist_segs = f["data/domainB/obs/robot0_eye_in_hand_segmentation_final"]  # (T, H, W)

    

        T = len(domainA_agentview_imgs)

        for t in tqdm(range(T)):
            idx_a = counters["agentview"]
            idx_w = counters["wristview"]

            # agentview
            # domainA
            Image.fromarray(domainA_agentview_imgs[t].astype(np.uint8)).save(
                os.path.join(out_agentview + "_domainA", f"{idx_a}.png")
            )
            np.save(os.path.join(out_agentview + "_domainA", f"{idx_a}_seg.npy"), domainA_agentview_segs[t].squeeze())

            # domainB
            Image.fromarray(domainB_agentview_imgs[t].astype(np.uint8)).save(
                os.path.join(out_agentview + "_domainB", f"{idx_a}.png")
            )
            np.save(os.path.join(out_agentview + "_domainB", f"{idx_a}_seg.npy"), domainB_agentview_segs[t].squeeze())
            counters["agentview"] += 1

            # wristview
            Image.fromarray(domainA_wrist_imgs[t].astype(np.uint8)).save(
                os.path.join(out_wristview + "_domainA", f"{idx_w}.png")
            )
            np.save(os.path.join(out_wristview + "_domainA", f"{idx_w}_seg.npy"), domainA_wrist_segs[t].squeeze())

            # domainB 
            Image.fromarray(domainB_wrist_imgs[t].astype(np.uint8)).save(
                os.path.join(out_wristview + "_domainB", f"{idx_w}.png")
            )
            np.save(os.path.join(out_wristview + "_domainB", f"{idx_w}_seg.npy"), domainB_wrist_segs[t].squeeze())
            counters["wristview"] += 1


def extract_from_file(hdf5_path, out_agentview, out_wristview, counters):
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        for demo in tqdm(demos, desc=f"Processing {os.path.basename(hdf5_path)}"):
            obs = f["data"][demo]["obs"]

            # Adjust keys if needed
            agentview_imgs = obs["agentview_image"]                # (T, H, W, 3)
            wrist_imgs = obs["robot0_eye_in_hand_image"]           # (T, H, W, 3)
            agentview_segs = obs["agentview_segmentation_final"]   # (T, H, W)
            wrist_segs = obs["robot0_eye_in_hand_segmentation_final"]  # (T, H, W)

            T = len(agentview_imgs)

            for t in range(T):
                idx_a = counters["agentview"]
                idx_w = counters["wristview"]

                # agentview
                Image.fromarray(agentview_imgs[t].astype(np.uint8)).save(
                    os.path.join(out_agentview, f"{idx_a}.png")
                )
                np.save(os.path.join(out_agentview, f"{idx_a}_seg.npy"), agentview_segs[t].squeeze())
                counters["agentview"] += 1

                # wristview
                Image.fromarray(wrist_imgs[t].astype(np.uint8)).save(
                    os.path.join(out_wristview, f"{idx_w}.png")
                )
                np.save(os.path.join(out_wristview, f"{idx_w}_seg.npy"), wrist_segs[t].squeeze())
                counters["wristview"] += 1


def main():
    parser = argparse.ArgumentParser(description="Extract images/segmentations from hdf5 files into flat folders")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing hdf5 files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for extracted data")
    parser.add_argument("--prefix", type=str, default="trainA", help="Prefix for output folders (default: trainA)")
    parser.add_argument("--test_data", action="store_true", help="Use for currating test dataset")
    args = parser.parse_args()

    out_agentview = os.path.join(args.output, f"{args.prefix}_agentview")
    out_wristview = os.path.join(args.output, f"{args.prefix}_wristview")
    if args.test_data:
        os.makedirs(out_agentview + "_domainA", exist_ok=True)
        os.makedirs(out_wristview + "_domainA", exist_ok=True) 
        os.makedirs(out_agentview + "_domainB", exist_ok=True)
        os.makedirs(out_wristview + "_domainB", exist_ok=True) 
    else:
        os.makedirs(out_agentview, exist_ok=True)
        os.makedirs(out_wristview, exist_ok=True)

    # Keep global counters so numbering continues across files/demos
    counters = {"agentview": 0, "wristview": 0}

    hdf5_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".hdf5")]
    if not hdf5_files:
        print(f"No hdf5 files found in {args.root}")
        return

    for hdf5_path in hdf5_files:
        if args.test_data:
            extract_test_data(hdf5_path, out_agentview, out_wristview, counters)
        else:
            extract_from_file(hdf5_path, out_agentview, out_wristview, counters)


if __name__ == "__main__":
    main()
