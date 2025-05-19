import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from openai import OpenAI
import base64
import io

from multiprocessing import Pool

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sandbox.MetaSandbox import GameSandbox, SandboxMananger
from sandbox.games import Game2048

import random
#set all random seed
np.random.seed(42)
random.seed(42)



def custom_policy(images_batch, prompt_batch):
    # you can implement your own policy here, return a batch of action strings
    return 

def random_policy(images_batch, prompt_batch):
    num_actions = 4
    return [ "<answer> " + str(np.random.randint(0, num_actions)) + " </answer>" for _ in range(len(images_batch))]


if __name__=="__main__":
    
    game = Game2048()

    SAVE_NAME = "test_2048"

    
    sandbox_list = []
    for i in range(1):
        sandbox_list.append(
            GameSandbox(
                make_env_func = game.gym_env_func,
                game_prompt = game.game_prompt,
                num_envs = 10,
                num_actions = game.num_actions,
                parse_action_func = game.parse_action_test_time,
                text_observation_func = game.get_text_observation,
                step_per_action = 1,
                episode_max_steps = 1000,
                screen_size = (640, 840),
            )
        )

    meta_sandbox = SandboxMananger(sandbox_list=sandbox_list)
    meta_sandbox.random_step_sandbox([100]*10)


    image_history = []
    response_history = []
    reward_history = []

    # meta_sandbox.random_step_sandbox([10, 20]) # 分别随机跑10步和20步, 目前2048没有加死亡判断

    for _ in tqdm(range(100)):
        online_experiences = meta_sandbox.step(random_policy)
        image_history.append([exp[0] for exp in online_experiences])
        response_history.append([exp[2] for exp in online_experiences])
        reward_history.append([exp[3] for exp in online_experiences])


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
    plt.title("2048 Culcumulated Reward of Each Run")
    plt.xlabel("Step")
    plt.ylabel("Culcumulated Reward")
    plt.savefig(f"logs/{SAVE_NAME}.png")
