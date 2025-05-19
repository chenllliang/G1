from .game import VGameEnv
import gymnasium as gym
import re


class GameBreakout(VGameEnv):
    def __init__(self):
        super().__init__()
        self.game_name = "Breakout"
        self.game_prompt = "You are now playing the Breakout game.\n\nAvailable actions are:\n- (0): Stay Still\n- (1): Release the ball (if there is no red ball on the screen)\n- (2): Move the paddle right\n- (3): Move the paddle left\n\nWhat action should you take to score given the current recent frame? Output the thinking process in <think></think> and action number in <answer></answer>."
        self.num_actions = 4

    def parse_action(self, response_text):
        # Extract the action from the response using regex
        action_match = re.search(r"<answer>.*?(\d+).*?</answer>", response_text, flags=re.DOTALL)
        if action_match:
            action = int(action_match.group(1))
            # Check if the action is within the valid range
            if action >= self.num_actions:
                print(f"Action {action} is out of range (0-{self.num_actions-1}). Defaulting to 0.")
                return 0
            return action
        else:
            print(response_text,"No valid action found in the response. Using 0")
            return 0

    def gym_env_func(self):
        return gym.make("ALE/Breakout-v5", render_mode="rgb_array")

