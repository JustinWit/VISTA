import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_frame_count(file, start_idx, n):
    with h5py.File(file) as f:
        frame_count = 0
        for i in range(n, start_idx):
            frame_count += f[f'data/demo_{i}/obs/agentview_image'].shape[0]
    
    frame_count *= 10

    return frame_count


def extract_from_file(hdf5_path, out_agentview, counters, num_demos, start_idx, view_per_frame):
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        demos.sort()
        demos = demos[start_idx:start_idx + num_demos]
        # shorten to only process specified demos
        for demo in tqdm(demos, desc=f"Processing {os.path.basename(hdf5_path)}"):
            obs = f["data"][demo]["obs"]

            
            agentview_imgs = obs[f"agentview_image"]
            T = len(agentview_imgs)

            for t in range(T):
                if view_per_frame == 1:
                    idx_a = counters["agentview"]
                    # idx_w = counters["wristview"]
                    # Adjust keys if needed
                    agentview_imgs = obs[f"agentview_image"]                     # (T, H, W, 3)
                    # wrist_imgs = obs[f"robot0_eye_in_hand_image"]                # (T, H, W, 3)
                    agentview_segs = obs[f"agentview_segmentation_final"]        # (T, H, W)
                    # wrist_segs = obs[f"robot0_eye_in_hand_segmentation_final"]   # (T, H, W)
                    # agentview
                    assert not os.path.exists(os.path.join(out_agentview, f"{idx_a}.png"))
                    assert not os.path.exists(os.path.join(out_agentview, f"{idx_a}_seg.npy"))
                    Image.fromarray(agentview_imgs[t].astype(np.uint8)).save(
                        os.path.join(out_agentview, f"{idx_a}.png")
                    )
                    np.save(os.path.join(out_agentview, f"{idx_a}_seg.npy"), agentview_segs[t].squeeze())
                    counters["agentview"] += 1

                    # wristview
                    # Image.fromarray(wrist_imgs[t].astype(np.uint8)).save(
                    #     os.path.join(out_wristview, f"{idx_w}.png")
                    # )
                    # np.save(os.path.join(out_wristview, f"{idx_w}_seg.npy"), wrist_segs[t].squeeze())
                    # counters["wristview"] += 1
                # else:
                #     for j in range(view_per_frame):
                #         idx_a = counters["agentview"]
                #         idx_w = counters["wristview"]
                #         # Adjust keys if needed
                #         agentview_imgs = obs[f"agentview_{j}_image"]                     # (T, H, W, 3)
                #         wrist_imgs = obs[f"robot0_eye_in_hand_{j}_image"]                # (T, H, W, 3)
                #         agentview_segs = obs[f"agentview_{j}_segmentation_final"]        # (T, H, W)
                #         wrist_segs = obs[f"robot0_eye_in_hand_{j}_segmentation_final"]   # (T, H, W)
                #         # agentview
                #         Image.fromarray(agentview_imgs[t].astype(np.uint8)).save(
                #             os.path.join(out_agentview, f"{idx_a}.png")
                #         )
                #         np.save(os.path.join(out_agentview, f"{idx_a}_seg.npy"), agentview_segs[t].squeeze())
                #         counters["agentview"] += 1

                #         # wristview
                #         Image.fromarray(wrist_imgs[t].astype(np.uint8)).save(
                #             os.path.join(out_wristview, f"{idx_w}.png")
                #         )
                #         np.save(os.path.join(out_wristview, f"{idx_w}_seg.npy"), wrist_segs[t].squeeze())
                #         counters["wristview"] += 1


def main():
    parser = argparse.ArgumentParser(description="Extract images/segmentations from hdf5 files into flat folders")
    parser.add_argument("--root", type=str, required=True, help="root with all hdf5 files to extract")
    parser.add_argument("--file", type=str, required=True, help="specific file to extract from")
    parser.add_argument("--output", type=str, required=True, help="Output folder for extracted data")
    parser.add_argument("--prefix", type=str, default="trainA", help="Prefix for output folders (default: trainA)")
    parser.add_argument("--n", type=int, default=None, help="Number of demos to process")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--views_per_frame", type=int, default=1, help="Specify views per frame")
    args = parser.parse_args()

    out_agentview = os.path.join(args.output, f"{args.prefix}_agentview")
    # out_wristview = os.path.join(args.output, f"{args.prefix}_wristview")
    os.makedirs(out_agentview, exist_ok=True)
    # os.makedirs(out_wristview, exist_ok=True)

    hdf5_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".hdf5")]
    hdf5_files.sort()
    if not hdf5_files:
        print(f"No hdf5 files found in {args.root}")
        return

    file_idx = hdf5_files.index(args.file)
    png_start_idx = 0
    # figure out start idx for png
    # first loop over all frames from previous tasks
    for file in range(file_idx):
        with h5py.File(hdf5_files[file], 'r') as f:
            demos = list(f['data'].keys())
            demos.sort()
            for i in demos:
                png_start_idx += (f[f'data/{i}/obs/agentview_image'].shape[0] * args.views_per_frame)
    # then loop over specified task up to the start_idx
    with h5py.File(args.file, 'r') as f:
        demos = list(f['data'].keys())
        demos.sort()
        for i in range(args.start_idx):
            png_start_idx += (f[f'data/{demos[i]}/obs/agentview_image'].shape[0] * args.views_per_frame)

    print(f'Start Idx: {png_start_idx}')

    # set counters
    # Keep global counters so numbering continues across files/demos
    counters = {"agentview": png_start_idx}

    extract_from_file(hdf5_files[file_idx], out_agentview, counters, args.n, args.start_idx, args.views_per_frame)


if __name__ == "__main__":
    main()
