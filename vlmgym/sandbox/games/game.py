class VGameEnv:
    """Base class for game environments with a standardized interface."""
    
    def __init__(self):
        """Initialize the game environment.
        
        Subclasses must set the following attributes:
        - game_name: Name of the game
        - game_prompt: Prompt text for the game
        - num_actions: Number of possible actions in the game
        """
        # These attributes must be set by subclasses
        self.game_name = None
        self.game_prompt = None
        self.num_actions = None
        
    
    def parse_action(self, action):
        """Parse model's string response into a game action.
        
        Args:
            action (str): The string response from the model
            
        Returns:
            The parsed action in the appropriate format for the game
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement parse_action method")

    def create_env_func(self):
        """Create and return a gym-like game environment.
        
        Returns:
            A gym-compatible environment with standard methods like reset() and step()
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_env_func method")
    
    def reward_shaping(self, reward):
        """Reward shaping function.
        
        Args:
            reward (float): The reward from the environment
            
        Returns:
            float: The shaped reward
        """
        return reward
    
    def get_text_observation(self, vec_env):
        """Get the text observation from the vectorized environment.
        
        Args:
            vec_env (VecEnv): The vectorized environment
            
        Returns:
            str: The text observation
        """
        raise NotImplementedError("Subclasses must implement get_text_observation method")
    

    
    

