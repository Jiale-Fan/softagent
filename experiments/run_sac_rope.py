from cem.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from cem.visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from softgym.utils.visualization import save_numpy_as_gif

# This script tries to run SAC on the SoftGym environments, with observation being key points.

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args


def get_env(vv, headless):
    env_name = vv['env_name']
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv = update_env_kwargs(vv)

    ###########################################
    vv["env_kwargs"]['num_variations'] = 84 # number of different initial states
    vv["env_kwargs"]['headless'] = headless
    # vv["env_kwargs"]['use_cached_states'] = False
    ###########################################

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': 200,
                  'action_repeat': 1,  # Action repeat for env wrapper is 1 as it is already inside the env
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    env = env_class(**env_kwargs)
    return env, vv


def sac_train(vv, log_dir, exp_name):
    
    mp.set_start_method('spawn')

    env, updated_vv = get_env(vv, headless=True)

    # Configure torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(vv['seed'])

    # Dump parameters
    logger.configure(dir=log_dir, exp_name=exp_name)
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(updated_vv, f, indent=2, sort_keys=True)

    # Check the environment complies with the gym interface
    # check_env(env._env._wrapped_env, warn=True, skip_render_check=True)

    policy_kwargs = dict(net_arch=[2048, 2048])
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=1000000, log_interval=4)
    model.save(vv["model_dir"])


def run_sac_policy(vv, log_dir, exp_name):
    mp.set_start_method('spawn')
    env, updated_vv = get_env(vv, headless=False)
    model = SAC("MlpPolicy", env, verbose=1)
    model.load(vv["model_dir"])

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Run policy
    initial_states, action_trajs, configs, all_infos = [], [], [], []

    env.reset() # remember to reset the env before getting the first image, otherwise there will be segmentation fault
    frames = [env.get_image(vv["img_size"], vv["img_size"])]
    for i in range(vv['test_episodes']):
        logger.log('episode ' + str(i))
        obs = env.reset() # if the obs mode is key_point, obs is a 36 dim vector containing coords of 10 key points and 2 gripper positions
        # policy.reset()
        initial_state = env.get_state()
        action_traj = []
        infos = []

        for j in range(env.horizon):
            logger.log('episode {}, step {}'.format(i, j))
            action = model.predict(obs, deterministic=True)[0]
            action_traj.append(copy.copy(action))
            obs, reward, _, info = env.step(action, record_continuous_video=True, img_size=vv["img_size"])
            infos.append(info)

            frames.extend(info['flex_env_recorded_frames'])
            
        all_infos.append(infos)
        initial_states.append(initial_state.copy())
        action_trajs.append(action_traj.copy())
        configs.append(env.get_current_config().copy())

        # Log for each episode
        transformed_info = transform_info([infos])
        for info_name in transformed_info:
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

        save_name = osp.join(vv["log_dir"], vv["env_name"] + f"_{i:01d}.gif")
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))

    # Dump trajectories
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='sac', type=str)
    parser.add_argument('--env_name', default='RopeFlatten', type=str)
    parser.add_argument('--log_dir', default='./data/sac', type=str)
    parser.add_argument('--model_dir', default='./sac_straighten_rope', type=str)
    parser.add_argument('--test_episodes', default=1, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--img_size', default=720, type=int)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)

    args = parser.parse_args()
    sac_train(args.__dict__, args.log_dir, args.exp_name)
    # run_sac_policy(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
