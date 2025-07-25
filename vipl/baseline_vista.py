from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
from vipl.utils.camera_pose_sampler import CameraPoseSampler
from vipl.utils.cam_utils import posori_to_rotmat
from vipl.utils.constants import ZERONVS_CHECKPOINT_PATH, ZERONVS_CONFIG_PATH

import robosuite.utils.transform_utils as T

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import numpy as np
import argparse
import os

def main(args):
    nvs_model = ZeroNVSModel(
                checkpoint=ZERONVS_CHECKPOINT_PATH,
                config=ZERONVS_CONFIG_PATH,
                zeronvs_params=dict(
                    ddim_steps=250,
                    ddim_eta=1.0,
                    precomputed_scale=0.6,
                    lpips_loss_threshold=0.7,
                    fov_deg=70,
                )
            )

    # this is what we use in RLBench as the initial camera pose
    initial_camera_matrix = np.array([
        [-1.43894450e-03, -5.77754720e-01, -8.16209172e-01,  1.15491505e+00],
        [-9.99977838e-01, -4.47417670e-03,  4.92997317e-03, -1.49455355e-02],
        [-6.50017933e-03,  8.16198178e-01, -5.77735478e-01,  1.56297836e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                        ])
    initial_camera_pose = T.mat2pose(initial_camera_matrix)  # convert to position and orientation


    random_camera_range = 'small_perturb'  # "how to sample the random camera poses. Options are 'small_perturb', 'arc_90deg'
    camera_pose_sampler = CameraPoseSampler(sampler_type=random_camera_range)

    # loop through the dataset
    fnames = sorted(os.listdir(args.data_path))
    fnames = [fname for fname in fnames if fname.endswith('.h5')]

    num_demos = len(fnames)

    for fname in tqdm(fnames):
        file_path = os.path.join(args.data_path, fname)
        data = h5py.File(file_path, 'r')

        third_ppov = data['rgb_frames'][:, 2]  # these are 640x360 and we need them to be 256x256
        third_ppov = third_ppov[:, :, 140:500]  # crop to 360x360
        third_ppov = third_ppov[:, :, :, ::-1]  # convert from BGR to RGB


        random_camera_poses = camera_pose_sampler.sample_poses(n=third_ppov.shape[0], starting_pose=initial_camera_pose)

        # Create a new dataset to store the augmented images
        augmented_frames = []

        for t in tqdm(range(third_ppov.shape[0])):
            # Use the ZeroNVS model to augment the image
            obs_augmented = nvs_model.augment(
                            original_image=Image.fromarray(third_ppov[t].astype(np.uint8)).resize((256, 256)),
                            original_camera=posori_to_rotmat(
                                position=initial_camera_pose[0],
                                orientation=initial_camera_pose[1],
                            ),  # OPTIONAL allegedly
                            target_camera=posori_to_rotmat(
                                position=random_camera_poses[t][0],
                                orientation=random_camera_poses[t][1],
                            ),
                            convention="opengl" # verified carefully that posori_to_rotmat returns cam2world in opengl convention
                        )

            # Save the augmented image back to the dataset
            cam1 = data['rgb_frames'][t, 0][:, 140:500]
            cam2 = data['rgb_frames'][t, 1][:, 140:500]
            cam1 = Image.fromarray(cam1.astype(np.uint8)).resize((256, 256))
            cam2 = Image.fromarray(cam2.astype(np.uint8)).resize((256, 256))
            augmented_frames.append(np.stack((cam1, cam2, np.array(obs_augmented)[:, :, ::-1]), axis=0))
            # breakpoint()
            # plt.imshow(np.hstack((third_ppov[t].astype(np.uint8), np.array(obs_augmented.resize((360, 360))))))
            # plt.axis('off')
            # plt.show()
            break

        # check if path exists, if not create it
        if not os.path.exists(args.data_path + "_vista"):
            os.makedirs(args.data_path + "_vista")
        
        demo_idx = int(fname.split('.')[0].split('_')[1]) + (num_demos * args.i)
        new_fname = f'demo_{demo_idx:03d}.h5'
        with h5py.File(os.path.join(args.data_path + "_vista", new_fname), 'w') as f:
            # copy original data
            for key in data.keys():
                if key != 'rgb_frames':
                    f.create_dataset(key, data=data[key])
            # create new dataset for augmented frames
            augmented_frames = np.array(augmented_frames)
            f.create_dataset('rgb_frames', data=augmented_frames)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZeroNVS augmentation")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('-i', type=int, required=False, help='Starting demo number to save file under. Use for array jobs', default=0)
    args = parser.parse_args()
    main(args)