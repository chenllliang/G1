from multiprocessing import Pool
import gymnasium as gym
import numpy as np
from typing import Callable, Dict, List
from tqdm import tqdm
import random
from PIL import Image
from copy import deepcopy



class SandboxMananger:
    """Manager class for handling multiple sandboxes (currently empty)"""
    def __init__(self, max_sample_number_in_queue: int=5000, sandbox_list: List['GameSandbox'] = None):
        """
        Initialize the SandboxMananger.
        
        Args:
            max_sample_number_in_queue: Maximum number of examples in the experience queue
            sandbox_list: List of GameSandbox objects, each sandbox is a batched game environment, such as [2048_1, 2048_2, 2048_3, Breakout_1, Breakout_2, Breakout_3]
        """
        self.max_sample_number_in_queue = max_sample_number_in_queue
        self.sandbox_list = sandbox_list if sandbox_list is not None else []
        self.experience_queue = ExperienceQueue(max_sample_number_in_queue)
        # each sample in the experience queue is a tuple of (image, prompt, response, reward)

        self.reset_sandbox()
    
    def random_step_sandbox(self, num_steps_per_sandbox: List, close_render: bool = False):
        """Randomly step each sandbox for a certain number of steps."""

        if close_render:
            for sandbox, num_steps in zip(self.sandbox_list, num_steps_per_sandbox):
                # print(sandbox.vec_envs.envs[0].close_render)
                for step in range(num_steps):
                    # print(f"{step}/{num_steps} steps: {close_render}")
                    if step < num_steps - 1 and close_render:
                        for env in sandbox.vec_envs.envs:
                            env.set_close_render(True) 
                    else:
                        for env in sandbox.vec_envs.envs:
                            env.set_close_render(False)
                    sandbox.step_random()
        else:

            for sandbox, num_steps in zip(self.sandbox_list, num_steps_per_sandbox):
                for step in range(num_steps):
                    sandbox.step_random()
        
        
        return

    def reset_sandbox(self):
        """Reset all batched sandboxes."""
        for sandbox in self.sandbox_list:
            sandbox.reset()

    def step(self, get_model_batched_response: Callable):
        """Step all batched sandboxes. Call the model to get the response, and add the sample to the experience queue.
        Args:
            get_model_batched_response: Function that takes a list of images and prompts, returns a list of responses
        """
        online_experiences = []
        for sandbox in self.sandbox_list:
            images, prompts, responses, rewards = sandbox.step_model(get_model_batched_response)
            # unbatched the return, add the sample to the experience queue
            for image, prompt, response, reward in zip(images, prompts, responses, rewards):
                online_experiences.append((image, prompt, response, reward))
        
        return online_experiences # also add some experience to the experience queue for replay, leave for now

    def get_image_prompt_batch(self):
        """Get the image and prompt batch for the current state of the sandbox."""
        current_image_batch = []
        current_prompt_batch = []
        for sandbox in self.sandbox_list:
            images, prompts = sandbox.get_image_prompt_batch()
            current_image_batch.extend(images)
            current_prompt_batch.extend(prompts)
        return current_image_batch, current_prompt_batch

    def get_reward_batch(self, model_batched_response: List[str]):
        """Get the reward batch for the current state of the sandbox."""
        # split the model_batched_response according to the num_envs in each sandbox
        rewards = []
        num_envs_in_each_sandbox = [sandbox.num_envs for sandbox in self.sandbox_list]

        # split the model_batched_response according to the num_envs in each sandbox
        batched_response_for_each_sandbox = []
        
        # Split responses for each sandbox based on their environment counts
        start_idx = 0
        for num_envs in num_envs_in_each_sandbox:
            end_idx = start_idx + num_envs
            sandbox_responses = model_batched_response[start_idx:end_idx]
            batched_response_for_each_sandbox.append(sandbox_responses)
            start_idx = end_idx
            
        # Get rewards from each sandbox using their responses
        for sandbox, sandbox_responses in zip(self.sandbox_list, batched_response_for_each_sandbox):
            sandbox_rewards = sandbox.step_env_with_batched_response(sandbox_responses)
            rewards.extend(sandbox_rewards)
            
        return rewards
    
    def get_perception_reward_batch(self, model_batched_response: List[str]):
        """Get the perception reward batch for the current state of the sandbox."""
        perception_rewards = []
        ground_truth_perception = self.get_text_observation_batch()
        
        for perception, response in zip(ground_truth_perception, model_batched_response):
            if perception in response:
                perception_rewards.append(1)
            else:
                perception_rewards.append(0)
        
        return perception_rewards


    def get_text_observation_batch(self):
        """Get the text observation batch for the current state of the sandbox."""
        current_text_observation_batch = []
        for sandbox in self.sandbox_list:
            current_text_observation_batch.extend(sandbox.get_text_observation())
        return current_text_observation_batch
    
    def get_reward_batch_with_n_rollouts(self, model_batched_response_rollouts: List[List[str]], n: int):
        """Get the reward batch for the current state of the sandbox."""
        rewards = []
        num_envs_in_each_sandbox = [sandbox.num_envs for sandbox in self.sandbox_list]

        # Create a deep copy of each board state for each environment
        init_board_states = []
        for sandbox in self.sandbox_list:
            # Get the board attribute from each environment
            boards = sandbox.vec_envs.get_attr("board")
            # Deep copy each board individually
            copied_boards = [deepcopy(board) for board in boards]
            init_board_states.append(copied_boards)

        for rollout_idx in range(n):
            rewards_per_rollout_index = []
            model_batched_response = model_batched_response_rollouts[rollout_idx]

            # Split responses for each sandbox based on their environment counts
            batched_response_for_each_sandbox = []
            start_idx = 0
            for num_envs in num_envs_in_each_sandbox:
                end_idx = start_idx + num_envs
                sandbox_responses = model_batched_response[start_idx:end_idx]
                batched_response_for_each_sandbox.append(sandbox_responses)
                start_idx = end_idx
                
            # Get rewards from each sandbox using their responses
            for sandbox, sandbox_responses in zip(self.sandbox_list, batched_response_for_each_sandbox):
                sandbox_rewards = sandbox.step_env_with_batched_response(sandbox_responses)
                rewards_per_rollout_index.extend(sandbox_rewards)
                
            rewards.append(rewards_per_rollout_index)

            # Reset each environment to its initial board state
            for sandbox_idx, sandbox in enumerate(self.sandbox_list):
                # Set each environment's board back to its initial state
                for env_idx, board in enumerate(init_board_states[sandbox_idx]):
                    sandbox.vec_envs.envs[env_idx].board = deepcopy(board)
            

        
        return rewards

    def get_reward_batch_with_n_rollouts_minesweeper(self, model_batched_response_rollouts: List[List[str]], n: int):
        """Get the reward batch for the current state of the sandbox."""
        rewards = []
        num_envs_in_each_sandbox = [sandbox.num_envs for sandbox in self.sandbox_list]
        # Create a deep copy of each board state for each environment
        init_board_states = []
        init_visible_states = []  # 新增: 保存visible状态
        for sandbox in self.sandbox_list:
            # Get the board attribute from each environment
            boards = sandbox.vec_envs.get_attr("board")
            visibles = sandbox.vec_envs.get_attr("visible")  # 新增: 获取visible属性
            # Deep copy each board individually
            copied_boards = [deepcopy(board) for board in boards]
            copied_visibles = [deepcopy(visible) for visible in visibles]  # 新增: 复制visible状态
            init_board_states.append(copied_boards)
            init_visible_states.append(copied_visibles)  # 新增: 保存visible状态
        for rollout_idx in range(n):
            rewards_per_rollout_index = []
            model_batched_response = model_batched_response_rollouts[rollout_idx]
            # Split responses for each sandbox based on their environment counts
            batched_response_for_each_sandbox = []
            start_idx = 0
            for num_envs in num_envs_in_each_sandbox:
                end_idx = start_idx + num_envs
                sandbox_responses = model_batched_response[start_idx:end_idx]
                batched_response_for_each_sandbox.append(sandbox_responses)
                start_idx = end_idx
            # Get rewards from each sandbox using their responses
            for sandbox, sandbox_responses in zip(self.sandbox_list, batched_response_for_each_sandbox):
                sandbox_rewards = sandbox.step_env_with_batched_response(sandbox_responses)
                rewards_per_rollout_index.extend(sandbox_rewards)
            rewards.append(rewards_per_rollout_index)
            # Reset each environment to its initial board state
            for sandbox_idx, sandbox in enumerate(self.sandbox_list):
                # Set each environment's board back to its initial state
                for env_idx, board in enumerate(init_board_states[sandbox_idx]):
                    sandbox.vec_envs.envs[env_idx].board = deepcopy(board)
                    sandbox.vec_envs.envs[env_idx].visible = deepcopy(init_visible_states[sandbox_idx][env_idx])  # 新增: 恢复visible状态
        return rewards
        
        
        














class ExperienceQueue:
    """A queue of experiences."""
    def __init__(self, max_sample_number_in_queue: int=5000):
        self.max_sample_number_in_queue = max_sample_number_in_queue
        self.experience_queue = []

    def add_sample(self, sample: tuple):
        """Add a sample to the experience queue. Remove the oldest sample if the queue is full."""
        if len(self.experience_queue) >= self.max_sample_number_in_queue:
            self.experience_queue.pop(0)
        self.experience_queue.append(sample)

    def random_get_n_samples(self, n: int):
        """Randomly get n samples from the experience queue."""
        return random.sample(self.experience_queue, n)

class GameSandbox:
    """A sandbox environment for running multiple game environments in parallel with AI agents."""
    
    def __init__(self, make_env_func: Callable, game_prompt: str, parse_action_func: Callable,
                 text_observation_func: Callable, num_envs: int = 8, num_actions: int = 1, step_per_action: int = 4, episode_max_steps: int = 1000, 
                 screen_size: tuple = (640, 840)):
        """
        Initialize the GameSandbox.
        
        Args:
            make_env_func: Function to create game environments
            game_prompt: Prompt for the game
            num_envs: Number of parallel environments
            num_actions: Number of valid actions, used to generate the action space and prune the unvalid output actions
            step_per_action: Number of steps between each action decision
            episode_max_steps: Maximum steps per episode, would reset the environment if the episode is over
            screen_size: Size of the game screen (width, height)
            save_experiencepath: Path to save experience mp4
        """

        self.num_envs = num_envs
        self.prompt = game_prompt
        self.vec_envs = gym.vector.SyncVectorEnv([make_env_func for _ in range(num_envs)])
        self.num_actions = num_actions
        self.screen_size = screen_size
        self.step_per_action = step_per_action
        self.episode_max_steps = episode_max_steps

        self.parse_action_func = parse_action_func

        self.text_observation_func = text_observation_func


        # print information of the sandbox
        print(f"\033[94mPrompt Template: {self.prompt}\033[0m")
        print(f"\033[96mNumber of Envs: {self.num_envs}\033[0m")

        self.experience_buffer = [[] for _ in range(self.num_envs)]
        self.frame_buffers = [[] for _ in range(self.num_envs)]
        self.observations, self.info = self.vec_envs.reset()

    
    def reset(self):
        """Reset all environments and return initial observations."""
        self.observations, self.info = self.vec_envs.reset()
        return
    
    def get_image_prompt_batch(self):
        """Get the image and prompt batch for the current state of the sandbox."""
        current_image_batch = []
        current_prompt_batch = []
        
        # Process each environment's current state
        for env_idx in range(self.num_envs):
            assert self.observations[env_idx] is not None
            # Convert observation to PIL image and resize
            img_array = self.observations[env_idx]
            pil_image = Image.fromarray(img_array) # 
            pil_image = pil_image.resize(self.screen_size, Image.Resampling.BICUBIC)
            
            current_image_batch.append(pil_image)
            current_prompt_batch.append(self.prompt)
            
        return current_image_batch, current_prompt_batch
    
    def step_env_with_batched_response(self, model_batched_response: List[str]):
        """Take a step in all environments using the provided model calling function."""
        actions = np.zeros(self.num_envs, dtype=int)
        for env_idx in range(self.num_envs):
            actions[env_idx] = self.parse_action_func(model_batched_response[env_idx])

        # Step environments and update rewards
        self.observations, rewards, terminateds, truncateds, infos = self.step_env(actions)

        # Step zero actions for self.step_per_action-1 steps
        actions = np.zeros(self.num_envs, dtype=int)
        for _ in range(self.step_per_action-1):
            self.observations, _, _, _, _ = self.step_env(actions)

        return rewards

    def step_env(self, actions: np.ndarray, close_render=False):
        """Take a step in all environments using the provided actions."""
        if close_render:
            return self.vec_envs.step(actions, close_render=close_render)
        else:
            return self.vec_envs.step(actions)
    

    def step_random(self, close_render: bool = False):
        """Take a random step in all environments"""
        actions = np.random.randint(0, self.num_actions, size=self.num_envs)
        if close_render:
             self.observations, rewards, terminateds, truncateds, infos = self.step_env(actions, close_render=close_render)
        else:
             self.observations, rewards, terminateds, truncateds, infos = self.step_env(actions)
        return

    
    def step_model(self, get_model_batched_response: Callable):
        """Take a step in all environments using the provided model calling function."""
        actions = np.zeros(self.num_envs, dtype=int)

        current_image_batch, current_prompt_batch = self.get_image_prompt_batch()
        
        model_batched_response = get_model_batched_response(current_image_batch, current_prompt_batch)
        for env_idx in range(self.num_envs):
            actions[env_idx] = self.parse_action_func(model_batched_response[env_idx])

        # Step environments and update rewards
        self.observations, rewards, terminateds, truncateds, infos = self.step_env(actions)

        # Step zero actions for self.step_per_action-1 steps
        actions = np.zeros(self.num_envs, dtype=int)
        for _ in range(self.step_per_action-1):
            self.observations, _, _, _, _ = self.step_env(actions)


        return current_image_batch, current_prompt_batch, model_batched_response, rewards

    def get_text_observation(self):
        """Get the text observation for the current state of the sandbox."""
        return self.text_observation_func(self.vec_envs)



