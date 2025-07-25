"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations.
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:

    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2

    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # extract 84x84 image and depth observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name depth.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --depth

    # (space saving option) extract 84x84 image observations with compression and without
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from PIL import Image

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

try:
    import mimicgen_envs
except Exception as e:
    print("You probably don't have mimicgen environments installed, which you might need :/")

from robosuite.utils.camera_utils import CameraMover, generate_random_camera_pose, generate_random_camera_pose_on_sphere, get_camera_extrinsic_matrix
from vipl.utils.env_utils import get_camera_info
from vipl.utils.cam_utils import posori_to_rotmat
from vipl.utils.camera_pose_sampler import CameraPoseSampler
from vipl.models.augmentation import get_model_by_name

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    camera_names=None,
    camera_height=84,
    camera_width=84,
    randomize_camera=False,
    camera_randomization_type=False,
    random_camera_range=None,
    nvs_model=None,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    for camera in camera_names:
        obs[camera + "_image"] = np.array(Image.fromarray(obs[camera + "_image"]).resize((camera_width, camera_height), resample=Image.LANCZOS))

    # maybe add in intrinsics and extrinsics for all cameras
    camera_info = list()

    is_robosuite_env = EnvUtils.is_robosuite_env(env=env)

    if is_robosuite_env and randomize_camera:
        camera_mover = CameraMover(
            env=env.env,
            camera="agentview"
        )

        initial_camera_pose = deepcopy(camera_mover.get_camera_pose())
        camera_pose_sampler = CameraPoseSampler(sampler_type=random_camera_range)
        random_camera_poses = camera_pose_sampler.sample_poses(n=states.shape[0], starting_pose=initial_camera_pose)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]

    # iteration variable @t is over "next obs" indices
    for t in tqdm(range(1, traj_len + 1)):
        if is_robosuite_env:
            # randomize camera position
            if randomize_camera and camera_randomization_type == "sim":
                pos, quat = random_camera_poses[t-1]
                camera_mover.set_camera_pose(
                    pos=pos,
                    quat=quat,
                )
            camera_info.append(get_camera_info(
                env=env,
                camera_names=camera_names,
                camera_height=camera_height,
                camera_width=camera_width,
            ))

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})
        if randomize_camera and camera_randomization_type != "sim":
        # if randomize_camera:
            # get zeronvs image
            next_obs_augmented = nvs_model.augment(
                original_image=Image.fromarray(next_obs["agentview_image"].astype(np.uint8)),
                original_camera=posori_to_rotmat(
                    position=initial_camera_pose[0],
                    orientation=initial_camera_pose[1],
                ),
                target_camera=posori_to_rotmat(
                    position=random_camera_poses[t-1][0],
                    orientation=random_camera_poses[t-1][1],
                ),
                convention="opengl" # verified carefully that posori_to_rotmat returns cam2world in opengl convention
            )
            next_obs["agentview_image"] = np.array(next_obs_augmented)
            for camera in camera_names:
                next_obs[camera + "_image"] = np.array(Image.fromarray(next_obs[camera + "_image"]).resize((camera_width, camera_height), resample=Image.LANCZOS))
        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj, camera_info


def get_demo_index_from_key(key):
    return int(key.split("_")[1])


def check_file(file_name, args, demos, all_demos):
    # check if the file already exists
    if not os.path.exists(file_name):
        return False
    # open the file
    try:
        f = h5py.File(file_name, "r", driver="core")
        for parse_iter in tqdm(range(args.parse_iters)):
            for ind in tqdm(range(len(demos))):
                # calculate the episode number based on the current iteration through the dataset and the demo iter
                # output_ind = parse_iter * len(all_demos) + ind
                # output_ep_name = "demo_{}".format(output_ind)
                # output_ep_name = demos[ind]
                # actual index of the demo in the original file, e.g. demo_6 -> 6.
                # This is not necessarily the same as the "ind" iterator
                demo_index = get_demo_index_from_key(demos[ind])
                output_ep_idx = parse_iter * len(all_demos) + demo_index
                output_ep_name = f"demo_{output_ep_idx}"
                # check if the episode exists
                if output_ep_name not in f["data"]:
                    f.close()
                    return False
                # make sure it's possible to actually open the demo, files can get corrupted at this level
                _ = f["data"][output_ep_name]
    except Exception as e:
        # if any error, file might be corrupted
        return False
    return True


def file_in_use(src):
    try:
        os.rename(src, src)
        return False
    except OSError:    # file is in use
        return True


def dataset_states_to_obs(args):
    if args.depth:
        assert len(args.camera_names) > 0, "must specify camera names if using depth"

    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    print(env_meta)
    camera_height = args.camera_height
    camera_width = args.camera_width

    if args.camera_randomization_type not in ["sim", "clone"]:
        camera_height = 256
        camera_width = 256

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=args.shaped,
        use_depth_obs=args.depth,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    all_demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = all_demos[args.start_idx:args.start_idx+args.n]
    else:
        demos = all_demos

    if args.start_idx != 0 or args.n is not None:
        print("==== Processing episodes {} to {} ====".format(args.start_idx, args.start_idx + len(demos)))
        args.output_name = "{}_{}-{}.hdf5".format(args.output_name[:-5], args.start_idx, args.start_idx + len(demos))

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)

    # check to see if the output file has already been created at that path and also has the desired number of episodes

    file_already_exists_and_valid = check_file(output_path, args, demos, all_demos)
    # in_use = file_in_use(output_path)
    try:
        x = h5py.File(output_path, "a")
        in_use = False
        x.close()
    except Exception as e:
        print(e)
        if isinstance(e, BlockingIOError):
            in_use = True
        else:
            in_use = False
    print("in use? ", in_use)
    if (file_already_exists_and_valid and not args.force_regenerate) or in_use:
        f.close()
        if file_already_exists_and_valid:
            print("File already exists and is valid, ending early!")
        elif in_use:
            print("File is in use, maybe being written by a different process...")
        if is_robosuite_env:
            env.env.close()
        exit()

    # call os remove to remove the file if it exists
    # this is to address some weird OSError no 5s
    if os.path.exists(output_path):
        os.remove(output_path)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    if args.camera_randomization_type != "sim":
        aug_model = get_model_by_name(args.camera_randomization_type)
    else:
        aug_model = None

    total_samples = 0
    for parse_iter in tqdm(range(args.parse_iters)):
        for ind in tqdm(range(len(demos))):
            # calculate the episode number based on the current iteration through the dataset and the demo iter
            # This is not necessarily the same as the "ind" iterator
            demo_index = get_demo_index_from_key(demos[ind])
            output_ep_idx = parse_iter * len(all_demos) + demo_index
            output_ep_name = f"demo_{output_ep_idx}"
            ep_to_read = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep_to_read)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep_to_read)].attrs["model_file"]

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep_to_read)][()]

            if args.randomize_cam:
                randomize_cam_range = args.randomize_cam_range
            else:
                randomize_cam_range = None

            traj, camera_info = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                randomize_camera=args.randomize_cam,
                camera_randomization_type=args.camera_randomization_type,
                random_camera_range=randomize_cam_range,
                nvs_model=aug_model,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep_to_read)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep_to_read)][()]

            # store transitions
            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(output_ep_name)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode

            if camera_info is not None:
                assert is_robosuite_env
                ep_data_grp.attrs["camera_info"] = json.dumps(camera_info, indent=4)

            total_samples += traj["actions"].shape[0]

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos) * args.parse_iters, output_path))
    f.close()
    f_out.close()
    if is_robosuite_env:
        env.env.close()


if __name__ == "__main__":

    import os
    # Check if the SLURM_JOB_ID environment variable is set and print it
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if slurm_job_id:
        print(f"SLURM Job ID: {slurm_job_id}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="start index to process trajectories",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )

    # flag for including depth observations per camera
    parser.add_argument(
        "--depth",
        action='store_true',
        help="(optional) use depth observations for each camera",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs",
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress",
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    # flag for randomizing camera positions
    parser.add_argument(
        "--randomize_cam",
        action='store_true',
        help="(optional) randomize camera positions at each step"
    )

    parser.add_argument(
        "--force_regenerate",
        action='store_true',
        help="(optional) don't allow early exit using existing data"
    )

    # randomization mode
    parser.add_argument(
        "--camera_randomization_type",
        type=str,
        default="sim",
        help="(optional) randomization mode for camera. Options are 'sim', 'zeronvs', 'deproj'"
    )

    parser.add_argument(
        "--parse-iters",
        type=int,
        default=1,
        help="""number of times to go over the dataset. This makes sense if you want to randomize camera positions,
             "and say, you want to randomize camera positions 10 times for each trajectory."""
    )

    # flag for randomizing camera angle std
    parser.add_argument(
        "--randomize_cam_range",
        type=str,
        default="arc_90deg",
        help="how to sample the random camera poses. Options are 'small_perturb', 'arc_90deg'"
    )

    parser.add_argument(
        "--zeronvs_scale",
        type=float,
        default=0.6,
        help="(optional) zeronvs precomputed scene scale"
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
