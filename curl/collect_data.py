import numpy as np
import torch
import os
import time
import json
import copy

from curl import utils
# from curl.logger import Logger

from curl.curl_sac import CurlSacAgent
from curl.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import Env

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt
from softgym.registered_env import env_arg_dict
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    main_run(vv_to_args(updated_vv))


def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            alpha_fixed=args.alpha_fixed,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main_run(args):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
        pre_aug_obs_shape = (3, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load(os.path.join(args.work_dir, 'model'), '150000')


    episode, episode_reward, done, ep_info = 0, 0, True, []
    start_time = time.time()

    img_size = 720
    
    sample_stochastically = False

    num_episodes = 100

    for j in range(10):

        all_frames, all_rewards, all_actions, all_keypoints, all_depths = [], [], [], [], []

        for i in tqdm(range(num_episodes)):
            obs = env.reset()
            done = False
            episode_reward = 0
            frames, rewards, actions, keypoints, depths = [], [], [], [], []
            while not done:
                keypoints.append(obs.to('cpu').numpy())
                frames.append(env.get_image(128, 128))
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                actions.append(action)
                obs, reward, done, info = env.step(action)
                depth = env.get_depth(128, 128)
                plt.imsave(arr=depth, fname="./data/test.png", cmap='gray')

                depths.append(depth)
                rewards.append(reward)
                
            
            all_frames.append(frames)
            all_rewards.append(rewards)
            all_actions.append(actions)
            all_keypoints.append(keypoints)
            all_depths.append(depths)

        ## not converting to numpy arrays helps to deal with variable length episodes
        # all_frames_numpy = np.array(all_frames)
        # all_rewards_numpy = np.array(all_rewards)
        # all_actions_numpy = np.array(all_actions)
        # all_keypoints_numpy = np.array(all_keypoints)

        # data = {'states': all_frames_numpy, 'rewards': all_rewards_numpy, 'actions': all_actions_numpy, 'keypoints': all_keypoints_numpy}
        data = {'states': all_frames, 'rewards': all_rewards, 'actions': all_actions, 'keypoints': all_keypoints, 'depths': all_depths}
        with open(os.path.join(args.work_dir, f'data_{j}.pkl'), 'wb') as f:
            pickle.dump(data, f)
            print('Data saved to %s' % os.path.join(args.work_dir, f'data_{j}.pkl'))

    pass
    # all_frames = np.array(all_frames).swapaxes(0, 1)
    # all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])


reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
    'ClothFold': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_lr_decay(env_name, obs_mode):
    if env_name == 'RopeFlatten' or (env_name == 'ClothFlatten' and obs_mode == 'cam_rgb'):
        return 0.01
    elif obs_mode == 'point_cloud':
        return 0.01
    else:
        return None


def get_actor_critic_lr(env_name, obs_mode):
    if env_name == 'ClothFold' or (env_name == 'RopeFlatten' and obs_mode == 'point_cloud'):
        if obs_mode == 'cam_rgb':
            return 1e-4
        else:
            return 5e-4
    if obs_mode == 'cam_rgb':
        return 3e-4
    else:
        return 1e-3


def get_alpha_lr(env_name, obs_mode):
    if env_name == 'ClothFold':
        return 2e-5
    else:
        return 1e-3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--exp_name', default='CURL_SAC', type=str)
    parser.add_argument('--env_name', default='RopeFlatten', type=str) # !!! Everything starts here
    parser.add_argument('--log_dir', default='./data/curl_v0.4/', type=str)
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--save_tb', default=False)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=True)
    parser.add_argument('--save_model', default=True)  # Save trained models

    # CURL
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']

    args = parser.parse_args()

    args.algorithm = 'CURL'

    # Set env_specific parameters

    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_actor_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]

    ################################
    args.env_kwargs['headless'] = True

    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
