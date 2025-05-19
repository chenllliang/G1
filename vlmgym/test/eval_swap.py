import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.MetaSandbox import GameSandbox, SandboxMananger
from sandbox.games import GameMatch3

import random


def custom_policy(images_batch, prompt_batch):
    # you can implement your own policy here, return a batch of action strings
    return 

# random policy
def call_random_action(image_path_lists, prompt_lists):
    """
    Generate random actions for the Match Game environment.
    Returns a list of response texts.
    """
    GRID_SIZE = 8
    if len(image_path_lists) != len(prompt_lists):
        raise ValueError("The number of images must match the number of prompts.")
    candidates = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
        ]
    response_texts = []
    for _ in range(len(image_path_lists)):
        pos1 = random.choice(candidates)
        pos2 = random.choice(candidates)
        response_text = f"<answer>({pos1[0]}, {pos1[1]}) ({pos2[0]}, {pos2[1]})</answer>"
        response_texts.append(response_text)
    return response_texts


def eval_reward_shaping(reward):
    return 1 if reward > 0 else 0

if __name__=="__main__":
    
    game = GameMatch3()

    SAVE_NAME = "test_swap"
 
    sandbox_macth_1 = GameSandbox(
        make_env_func = game.gym_env_func,
        game_prompt = game.game_prompt,
        num_envs = 5,
        num_actions = game.num_actions,
        parse_action_func = game.parse_action,
        text_observation_func = game.get_text_observation,
        step_per_action = 1,
        episode_max_steps = 1000,
        screen_size = (640, 640)
    )

    sandbox_match_2 = GameSandbox(
        make_env_func = game.gym_env_func,
        game_prompt = game.game_prompt,
        num_envs = 5,
        num_actions = game.num_actions,
        parse_action_func = game.parse_action,
        text_observation_func = game.get_text_observation,
        step_per_action = 1,
        episode_max_steps = 1000,
        screen_size = (640, 640),
    )

    meta_sandbox = SandboxMananger(sandbox_list=[sandbox_macth_1, sandbox_match_2])


    image_history = []
    response_history = []
    reward_history = []


    for _ in tqdm(range(100)):
        online_experiences = meta_sandbox.step(call_random_action)
        image_history.append([exp[0] for exp in online_experiences])
        response_history.append([exp[2] for exp in online_experiences])
        reward_history.append([eval_reward_shaping(exp[3]) for exp in online_experiences])


    # reschedule the history list, make the same index of different batch in the same list
    image_history = list(map(list, zip(*image_history)))
    reward_history = list(map(list, zip(*reward_history)))
    response_history = list(map(list, zip(*response_history)))

    # combine the reward and response history
    combined_history = []
    for reward, response in zip(reward_history, response_history):
        combined_history.append((reward, response))
    
    os.makedirs("logs", exist_ok=True)

    # save the combined history to json
    with open(f"logs/{SAVE_NAME}.json", "w") as f:
        json.dump(combined_history, f, indent=4)


    # make the videos folder
    os.makedirs("videos", exist_ok=True)

    for idx, image_batch in enumerate(image_history):
        imageio.mimsave(f"videos/{SAVE_NAME}_{idx}.mp4", [np.array(img) for img in image_batch], fps=30, codec="libx264")
    
    plt.figure(dpi=200)
    for idx, reward_batch in enumerate(reward_history):
        plt.plot(np.cumsum(reward_batch), label=f"Run {idx}")
    
    plt.legend(loc="upper left")
    plt.title("Swap Culcumulated Reward of Each Run")
    plt.xlabel("Step")
    plt.ylabel("Culcumulated Reward")
    plt.savefig(f"logs/{SAVE_NAME}.png")
