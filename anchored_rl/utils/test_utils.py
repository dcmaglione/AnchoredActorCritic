import pickle
from anchored_rl.utils import save_utils, loss_composition
import numpy as np
import os


def test(actor, env, seed=123, render=True, num_steps=400):
    o,i = env.reset(seed=seed)
    high = env.action_space.high
    low = env.action_space.low
    os = []
    rs = []
    for _ in range(num_steps):
        o, r, d, _, i, = env.step(actor(o)*(high - low)/2.0 + (high + low)/2.0)
        if d:
            break
        os.append(o)
        rs.append(r)
        if render:
            env.render()
    print("reward sum:", np.sum(rs))
    return np.array(os), np.array(rs)

def folder_to_episode_rewards(env, render, num_tests, folder_path, steps=400, seed=123,  **kwargs):
    saved = save_utils.load_actor(folder_path)
    def actor(x):
        # print(np.array([x], dtype=np.float32))
        return saved(np.array([x], dtype=np.float32))[0]
    runs = list(map(lambda i: test(actor, env, seed=seed+i,
                    render=render, num_steps=steps)[1], range(num_tests)))
    
    return runs

def run_tests(env, cmd_args):
    print(cmd_args.save_folders)
    if cmd_args.use_latest:
        folders = [save_utils.latest_train_folder(folder) for folder in cmd_args.save_folders]
    else:
        folders = save_utils.concatenate_lists([save_utils.find_all_train_paths(folder) for folder in cmd_args.save_folders])
    print("################################")
    print("################################")
    print("################################")
    means = []
    data = {}
    for folder in folders:
        print("using folder:", folder.parent)
        episode_rewards = [np.sum(rewards) for rewards in folder_to_episode_rewards(env, folder_path=folder, **vars(cmd_args))]
        mean_reward = np.mean(episode_rewards)
        means.append(mean_reward)
        std = np.std(episode_rewards)
        data[folder] = episode_rewards
        print(f"{mean_reward:.4f}+-{std:.4f}")
        print(f"geomean: {loss_composition.geo(episode_rewards):.4f}")

    if hasattr(cmd_args, 'store_results') and cmd_args.store_results is not None:
        os.makedirs(os.path.dirname(cmd_args.store_results), exist_ok=True)
        with open(cmd_args.store_results, 'wb') as f:
            pickle.dump(data, f)
    print("################################")
    print("################################")
    print("################################")
    print(f"{np.mean(means):.4f}+-{np.std(means):.4f}")
