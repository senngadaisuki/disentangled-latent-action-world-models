import argparse
import os

from vp2.envs.robodesk_env import RoboDeskEnv
from vp2.envs.scripted_policies.robodesk_scripted_policies import *
from vp2.envs.scripted_policies.noisy_policy_wrapper import (
    NoisyPolicyWrapper,
)


BASE_DATA_DIR = "/home/xiaoliu/work/vp2/vp2/vp2_benchmark_data/robodesk_benchmark_tasks"

def main(args):
    # <--- 2. 手动构造 goals.hdf5 的绝对路径
    # 注意：这里我们用 f-string 动态填入 task 名字，这相当于手动做了 Hydra 该做的事
    goal_path = os.path.join(BASE_DATA_DIR, f"robodesk_{args.task}", "goals.hdf5")

    print(f"DEBUG: Manually set goal path to: {goal_path}") # 打印一下确认路径对不对

    # <--- 3. 在这里加上 goals_dataset 参数
    env = RoboDeskEnv(
        action_repeat=50, 
        image_size=256, 
        task=args.task, 
        reward="dense",
        goals_dataset=goal_path  # <--- 新增这行！
    )
    
    policy = TASK_TO_POLICY[args.task]
    policy = NoisyPolicyWrapper(policy, noise_std=0.4)
    run_traj(env, policy)


def run_traj(env, policy):
    env.reset()
    policy.reset()
    vis_frames = []
    rewards = []
    for i in range(30):
        action = policy.get_action(env)
        obs, rew, done, info = env.step(action)
        vis_frames.append(obs["rgb"])
        rewards.append(rew)
    print(f"Total reward is {sum(rewards)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str, default="push_red")
    args = args.parse_args()
    main(args)
