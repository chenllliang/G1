from .game import VGameEnv
import gymnasium as gym
import re
import numpy as np


class Game2048(VGameEnv):
    def __init__(self):
        super().__init__()
        self.game_name = "2048"
        self.game_prompt = "You are now playing the 2048 game. 2048 is a sliding tile puzzle game where you combine numbered tiles to create a tile with the value 2048.\n\nRule:\nOnly Tiles with the SAME number merge when they collide. After each move, a new tile (2 or 4) appears randomly on the board. The game ends when there are no more valid moves.\n\nAvailable actions:\n- (0): Up (slide all tiles upward)\n- (1): Right (slide all tiles to the right)\n- (2): Down (slide all tiles downward)\n- (3): Left (slide all tiles to the left)\nWhat action should you take to achieve the highest score and reach the 2048 tile?\n\nFirst describe the board in <perception></perception>.Then output your thinking process in <think></think> and final action number in <answer></answer>."
        
        #self.game_prompt = "You are now playing the 2048 game. 2048 is a sliding tile puzzle game where you combine numbered tiles to create a tile with the value 2048.\n\nRule:\nOnly Tiles with the SAME number merge when they collide. After each move, a new tile (2 or 4) appears randomly on the board. The game ends when there are no more valid moves.\n\nAvailable actions:\n- (0): Up (slide all tiles upward)\n- (1): Right (slide all tiles to the right)\n- (2): Down (slide all tiles downward)\n- (3): Left (slide all tiles to the left)\nWhat action should you take to achieve the highest score and reach the 2048 tile?\n\nFirst describe the board in <perception></perception>. Then output your thinking process (analyze the outcome of each action and choose the best one) in <think></think> and final action number in <answer></answer>."
        
        self.num_actions = 5


    def parse_action(self, response_text):
        # Extract the action from the response using regex
        action_match = re.search(r"<answer>.*?(\d+).*?</answer>", response_text, flags=re.DOTALL)
        if action_match:
            action = int(action_match.group(1))
            # Check if the action is within the valid range
            if action >= self.num_actions-1:
                print(f"Action {action} is out of range (0-{self.num_actions-2}). Defaulting to 4: Still.")
                return 4
            return action
        else:
            print("No valid action found in the response. Using 4: Still",response_text.replace("\n", " "))
            return 4
        
    def parse_action_test_time(self, response_text):
        # Extract the action from the response using regex, use random 0-3 if no action is found
        action_match = re.search(r"<answer>.*?(\d+).*?</answer>", response_text, flags=re.DOTALL)
        if action_match:
            action = int(action_match.group(1))
            # Check if the action is within the valid range
            if action >= self.num_actions-1:
                print(f"Action {action} is out of range (0-{self.num_actions-2}). Defaulting to random action")
                return np.random.randint(0, self.num_actions-1)
            return action
        else:
            print("No valid action found in the response. Using random action")
            return np.random.randint(0, self.num_actions-1)

    def gym_env_func(self):
        return gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", render_mode="rgb_array")

    def reward_shaping(self, reward):
        """
        0->0, 8->3, 16->4, 32->5, 64->6, 128->7, 256->8, 512->9, 1024->10, 2048->11, 4096->12, 8192->13, 16384->14, 32768->15, 65536->16
        """
        return 1 if reward > 0 else -1
    
    def reward_shaping_01(self, reward):

        return 1 if reward > 0 else 0
    
    def state_to_text(self, state):
        """
        Convert the state to a text observation. Each number is replaced by 2^number. If the state is 0, it is replaced by empty.
        """
        return "\n".join(["|".join([" "+str(2**cell)+" " if cell != 0 else " - " for cell in row]) for row in state])



    def get_text_observation(self, vec_env):
        """
        Get the text observation from the vectorized environment. The 
        """
        
        states = [ i for i in vec_env.get_attr("board")]

        text_observations = [self.state_to_text(state) for state in states]

        return text_observations

