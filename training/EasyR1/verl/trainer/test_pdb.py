

from datasets import load_dataset


from r1_gamer.sandbox.games import Game2048, GameBreakout
from r1_gamer.sandbox.MetaSandbox import SandboxMananger, GameSandbox, ExperienceQueue

import datasets
from datasets import Dataset, Features, Value, Image


game = Game2048()

sandbox_2048_1 = GameSandbox(
    make_env_func = game.gym_env_func,
    game_prompt = game.game_prompt,
    num_envs = 5,
    num_actions = game.num_actions,
    parse_action_func = game.parse_action,
    step_per_action = 1,
    episode_max_steps = 1000,
    screen_size = (640, 840)
)

sandbox_2048_2 = GameSandbox(
    make_env_func = game.gym_env_func,
    game_prompt = game.game_prompt,
    num_envs = 5,
    num_actions = game.num_actions,
    parse_action_func = game.parse_action,
    step_per_action = 1,
    episode_max_steps = 1000,
    screen_size = (640, 840),
)
# 如果 step_per_action不等于1，reward 会消失



for i in range(10):
    
    

    META_SANDBOX = SandboxMananger(sandbox_list=[sandbox_2048_1, sandbox_2048_2])

    def make_huggingface_dataset_from_image_prompt_batch(images, prompts):
        # the dataset has two columns: problem and images. The problem is a string, add "<image>" to the front. The images column is a list of PIL images.

        features = Features(
            {'problem': Value(dtype='string', id=None),
            'images': datasets.Sequence(Image())})
        
        # Prepare data
        data = {
            'problem': ["<image>" + prompt for prompt in prompts],
            'images': [[image] for image in images]
        }
        
        # Create and return the dataset
        return Dataset.from_dict(data, features=features)

    print("hello")