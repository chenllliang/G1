import base64
import json
import os
import random
import re
import time
import math
from collections import deque
from multiprocessing import Pool

import gymnasium as gym
import imageio
import numpy as np
import pygame
from tqdm import tqdm

# ----------------- Constants Definition -----------------
# Match-3 Game configuration parameters
GRID_SIZE = 8          # Board size is 8x8
TILE_SIZE = 70         # Each tile is 70 pixels in size
SCREEN_WIDTH = GRID_SIZE * TILE_SIZE
SCREEN_HEIGHT = GRID_SIZE * TILE_SIZE + 60  # Additional 60 pixels for score display
FPS = 60

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Semi-transparent yellow

# Reduced number of colors
TILE_COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
]

# Reduced number of shapes
SHAPES = ["circle", "square", "triangle"]

TILE_COLORS_MATCH = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
}


# ----------------- Match-3 Game Gym Environment -----------------
class Match3GameEnv(gym.Env):
    """
    Gymnasium environment for a Match-3 Game.
    Observation: RGB image of the screen.
    Action: (row1, col1, row2, col2) representing the two adjacent tiles to swap.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Match-3 Game")
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Define action_space as a 4-dimensional discrete space
        self.action_space = gym.spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE])
        
        # Observation: RGB array of the screen image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), 
            dtype=np.uint8
        )
        
        self._init_game()

    def _init_game(self):
        """Initialize the game state."""
        self.board = self.create_board()
        self.selected_tile = None
        self.game_over = False
        self.score = 0
        self.moves_left = 1000000
        self.step_count = 0
        self.message = None     # For displaying game messages
        self.message_timer = 0  # Timer for message display
        self.shuffle_count = 0  # Just for tracking, no limit
        
        # Ensure no matches at game start
        while self.find_matches():
            self.remove_matches(self.find_matches())
            self.fill_empty_spaces()
        
        # Ensure the board has valid moves
        while not self.has_valid_moves():
            self.shuffle_board()
        
    def create_board(self):
        """Generate the board with random tiles."""
        board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Pick a random color and shape
                color = random.choice(TILE_COLORS)
                shape = random.choice(SHAPES)
                board[i][j] = (color, shape)
                
                # Avoid creating matches at board creation
                attempts = 0
                while attempts < 10 and self.would_create_match(board, i, j):
                    color = random.choice(TILE_COLORS)
                    shape = random.choice(SHAPES)
                    board[i][j] = (color, shape)
                    attempts += 1
                    
        return board
    
    def would_create_match(self, board, row, col):
        """Check if a tile would create a match if placed at (row, col)."""
        if row < 2:  # Not enough tiles above to check for a vertical match
            vertical_match = False
        else:
            vertical_match = (board[row-1][col] == board[row-2][col] == board[row][col])
            
        if col < 2:  # Not enough tiles to the left to check for a horizontal match
            horizontal_match = False
        else:
            horizontal_match = (board[row][col-1] == board[row][col-2] == board[row][col])
            
        return vertical_match or horizontal_match
    
    def draw_shape(self, surface, shape, color, rect):
        """Draw the specified tile shape with the given color in the rectangular area."""
        x, y, w, h = rect
        padding = 10
        inner_rect = (x + padding, y + padding, w - 2 * padding, h - 2 * padding)
        
        if shape == "circle":
            pygame.draw.ellipse(surface, color, inner_rect)
            pygame.draw.ellipse(surface, BLACK, inner_rect, 2)
        elif shape == "square":
            pygame.draw.rect(surface, color, inner_rect)
            pygame.draw.rect(surface, BLACK, inner_rect, 2)
        elif shape == "triangle":
            points = [
                (x + w // 2, y + padding),
                (x + padding, y + h - padding),
                (x + w - padding, y + h - padding)
            ]
            pygame.draw.polygon(surface, color, points)
            pygame.draw.polygon(surface, BLACK, points, 2)

    def draw_board(self):
        """Draw the entire board, all tiles, and game information on the screen."""
        self.screen.fill(WHITE)
        
        # Draw grid background
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                bg_color = (240, 240, 240) if (i + j) % 2 == 0 else (230, 230, 230)
                pygame.draw.rect(self.screen, bg_color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Draw tiles
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.board[i][j] is not None:
                    rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    
                    # Draw shadow effect
                    shadow_offset = 4
                    shadow_rect = pygame.Rect(
                        rect.left + shadow_offset, 
                        rect.top + shadow_offset,
                        rect.width - shadow_offset, 
                        rect.height - shadow_offset
                    )
                    pygame.draw.rect(self.screen, (100, 100, 100, 128), shadow_rect)
                    
                    # Draw tile background
                    tile_padding = 2
                    tile_rect = pygame.Rect(
                        rect.left + tile_padding,
                        rect.top + tile_padding,
                        rect.width - tile_padding * 2,
                        rect.height - tile_padding * 2
                    )
                    pygame.draw.rect(self.screen, (245, 245, 245), tile_rect)
                    pygame.draw.rect(self.screen, (180, 180, 180), tile_rect, 1)
                    
                    # Draw the shape on the tile
                    color, shape = self.board[i][j]
                    self.draw_shape(self.screen, shape, color, tile_rect)
        
        # Draw selection highlight
        if self.selected_tile:
            i, j = self.selected_tile
            highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            highlight.fill(HIGHLIGHT_COLOR)
            self.screen.blit(highlight, (j * TILE_SIZE, i * TILE_SIZE))
        
        # Draw score and moves left
        font = pygame.font.SysFont(None, 32)
        score_text = font.render(f"Score: 0   Moves: 1000000   Shuffles: 0", True, BLACK)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 40))
        
        # Draw game message if present
        if self.message and self.message_timer > 0:
            msg_font = pygame.font.SysFont(None, 42)
            msg_text = msg_font.render(self.message, True, (255, 0, 0))
            text_rect = msg_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
            self.screen.blit(msg_text, text_rect)
            self.message_timer -= 1
        
        if self.game_over:
            game_over_font = pygame.font.SysFont(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        if self.render_mode == "human":
            pygame.display.flip()

    def find_matches(self):
        """Find all matching groups of 3 or more tiles."""
        matches = []
        
        # Check horizontal matches
        for i in range(GRID_SIZE):
            j = 0
            while j < GRID_SIZE - 2:
                if self.board[i][j] is not None:
                    current_tile = self.board[i][j]
                    match_len = 1
                    k = j + 1
                    while k < GRID_SIZE and self.board[i][k] == current_tile:
                        match_len += 1
                        k += 1
                    
                    if match_len >= 3:
                        matches.extend([(i, j + m) for m in range(match_len)])
                        j = k
                    else:
                        j += 1
                else:
                    j += 1
        
        # Check vertical matches
        for j in range(GRID_SIZE):
            i = 0
            while i < GRID_SIZE - 2:
                if self.board[i][j] is not None:
                    current_tile = self.board[i][j]
                    match_len = 1
                    k = i + 1
                    while k < GRID_SIZE and self.board[k][j] == current_tile:
                        match_len += 1
                        k += 1
                    
                    if match_len >= 3:
                        matches.extend([(i + m, j) for m in range(match_len)])
                        i = k
                    else:
                        i += 1
                else:
                    i += 1
        
        return list(set(matches))  # Remove duplicates
    
    def remove_matches(self, matches):
        """Remove matched tiles and add score."""
        if not matches:
            return False
            
        for row, col in matches:
            self.board[row][col] = None
        
        # Score based on number of tiles removed
        if len(matches) >= 3:
            match_bonus = len(matches) - 2  # Bonus for matches longer than 3
            self.score += len(matches) * (1 + match_bonus)
            
        return True
    
    def fill_empty_spaces(self):
        """Fill empty spaces by moving tiles down and generating new ones at the top."""
        # For each column, move tiles down to fill empty spaces
        for col in range(GRID_SIZE):
            # Count empty spaces from bottom to top
            for row in range(GRID_SIZE-1, -1, -1):
                if self.board[row][col] is None:
                    # Find the nearest non-empty tile above
                    for above_row in range(row-1, -1, -1):
                        if self.board[above_row][col] is not None:
                            # Move the tile down
                            self.board[row][col] = self.board[above_row][col]
                            self.board[above_row][col] = None
                            break
        
        # Fill remaining empty spaces at the top with new tiles
        for col in range(GRID_SIZE):
            for row in range(GRID_SIZE):
                if self.board[row][col] is None:
                    color = random.choice(TILE_COLORS)
                    shape = random.choice(SHAPES)
                    self.board[row][col] = (color, shape)
    
    def is_valid_swap(self, pos1, pos2):
        """Check if the two positions are adjacent."""
        row1, col1 = pos1
        row2, col2 = pos2
        
        # Must be adjacent (not diagonal)
        adjacent = (abs(row1 - row2) == 1 and col1 == col2) or (abs(col1 - col2) == 1 and row1 == row2)
        
        # Also check if both positions are valid
        valid_pos1 = 0 <= row1 < GRID_SIZE and 0 <= col1 < GRID_SIZE
        valid_pos2 = 0 <= row2 < GRID_SIZE and 0 <= col2 < GRID_SIZE
        
        return adjacent and valid_pos1 and valid_pos2
    
    def swap_tiles(self, pos1, pos2):
        """Swap two tiles and check if it creates a match."""
        if not self.is_valid_swap(pos1, pos2):
            return False
        
        # Perform the swap
        row1, col1 = pos1
        row2, col2 = pos2
        self.board[row1][col1], self.board[row2][col2] = self.board[row2][col2], self.board[row1][col1]
        
        # Check if the swap created a match
        matches = self.find_matches()
        if matches:
            return True
        else:
            # If no match was created, swap back
            self.board[row1][col1], self.board[row2][col2] = self.board[row2][col2], self.board[row1][col1]
            return False
    
    def update_game_state(self):
        """Update the game state after a move."""
        # Find and remove matches, then fill empty spaces
        match_found = self.remove_matches(self.find_matches())
        
        if match_found:
            # Fill empty spaces
            self.fill_empty_spaces()
            
            # Check for cascading matches
            while self.find_matches():
                match_found = self.remove_matches(self.find_matches())
                if match_found:
                    # Render between each match for better visualization
                    if self.render_mode == "human":
                        self.render()
                        pygame.time.delay(300)  # Short delay to visualize the cascading effect
                    self.fill_empty_spaces()
        
        # Check if there are any valid moves left after the update
        if not self.has_valid_moves():
            self.shuffle_board()
            self.shuffle_count += 1
            self.set_message("No moves available - Board shuffled!")
    
    def has_valid_moves(self):
        """Check if there are any valid moves left."""
        # Check if there are any possible matching moves
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Try swapping with right neighbor
                if j < GRID_SIZE - 1:
                    self.board[i][j], self.board[i][j+1] = self.board[i][j+1], self.board[i][j]
                    if self.find_matches():
                        self.board[i][j], self.board[i][j+1] = self.board[i][j+1], self.board[i][j]
                        return True
                    self.board[i][j], self.board[i][j+1] = self.board[i][j+1], self.board[i][j]
                
                # Try swapping with bottom neighbor
                if i < GRID_SIZE - 1:
                    self.board[i][j], self.board[i+1][j] = self.board[i+1][j], self.board[i][j]
                    if self.find_matches():
                        self.board[i][j], self.board[i+1][j] = self.board[i+1][j], self.board[i][j]
                        return True
                    self.board[i][j], self.board[i+1][j] = self.board[i+1][j], self.board[i][j]
        
        return False  # No valid moves left
    
    def shuffle_board(self):
        """Shuffle the board while ensuring there are valid moves."""
        # Create a list of all tiles
        all_tiles = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                all_tiles.append(self.board[i][j])
        
        # Shuffle the list
        random.shuffle(all_tiles)
        
        # Place shuffled tiles back on the board
        index = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.board[i][j] = all_tiles[index]
                index += 1
        
        # Ensure no matches at start
        while self.find_matches():
            self.remove_matches(self.find_matches())
            self.fill_empty_spaces()
        
        # If still no valid moves, reshuffle
        if not self.has_valid_moves():
            self.shuffle_board()  # Recursive call to try again
    
    def set_message(self, text):
        """Set a message to display on the screen."""
        self.message = text
        self.message_timer = 0  # Display for 60 frames (about 1 second at 60 FPS)
    
    def check_game_over(self):
        """Check if moves_left is 0."""
        return self.moves_left <= 0
    
    def step(self, action):
        """
        Execute one move based on the action = (row1, col1, row2, col2).
        Returns (observation, reward, done, truncated, info).
        """
        if self.game_over:
            return (self.render(mode="rgb_array"), 0, True, False, 
                    {"score": self.score, "moves_left": self.moves_left, "shuffle_count": self.shuffle_count})
        
        # Parse the action
        row1 = action // (GRID_SIZE*GRID_SIZE*GRID_SIZE)
        col1 = (action % (GRID_SIZE*GRID_SIZE*GRID_SIZE)) // (GRID_SIZE*GRID_SIZE)
        row2 = (action % (GRID_SIZE*GRID_SIZE)) // GRID_SIZE
        col2 = action % GRID_SIZE
        
        pos1 = (row1, col1)
        pos2 = (row2, col2)
        
        # Store previous score to calculate reward
        prev_score = self.score
        
        # Try to swap the tiles
        success = self.swap_tiles(pos1, pos2)
        
        if success:
            # Update game state after a successful swap
            self.update_game_state()
            reward = self.score - prev_score  # Reward is the score gained in this move
            self.moves_left -= 1
        else:
            reward = -1  # Penalty for invalid move
            self.set_message("Invalid move!")
        
        # Check if game is over (only when moves left = 0)
        self.game_over = self.check_game_over()
        if self.game_over:
            self.set_message(f"Game Over! Final Score: {self.score}")
        
        # Update step count
        self.step_count += 1
        
        # Prepare info dict
        info = {
            "score": self.score, 
            "moves_left": self.moves_left,
            "shuffle_count": self.shuffle_count
        }
        
        # Get observation
        obs = self.render(mode="rgb_array")
        
        # If game is over in gym environment, we typically reset automatically
        if self.game_over and self.render_mode != "human":
            obs, info = self.reset()
        
        return obs, reward, self.game_over, False, info
    
    def reset(self, seed=None, options=None):
        """Reset the game to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._init_game()
        return self.render(mode="rgb_array"), {}
    
    def render(self, mode="rgb_array"):
        """Render the game state."""
        if mode is None:
            mode = self.render_mode
            
        self.draw_board()
        
        if mode == "human":
            self.clock.tick(FPS)
            return None
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        
        return None
    
    def close(self):
        """Close the environment."""
        pygame.quit()


# ----------------- Game Environment Wrapper -----------------
class GameMatch3:
    """
    Wrapper for the Match-3 Game environment with additional game-specific utilities.
    """
    def __init__(self):
        self.game_name = "Match-3 Game"
        self.game_prompt = """
        You are playing a Match-3 Game where you need to swap adjacent tiles to create matches of 3 or more identical tiles.
        
        Rules:
        - Tiles are identified by color (Red, Green, Blue, Yellow) and shape (circle, square, triangle)
        - You can only swap adjacent tiles (not diagonal)
        - A valid move must create at least one match of 3 or more identical tiles
        - After matches are removed, tiles above will fall down and new tiles will appear at the top
        - If no valid moves are available, the board will automatically be shuffled
        - The game ends when you run out of moves
        
        Please analyze the game board and identify two adjacent tiles to swap that will create a match.
        Return your answer as follows:
        1. First coordinate: (row1, col1)
        2. Second coordinate: (row2, col2)
        
        Where row and col are 0-indexed numbers starting from the top-left of the board.
        First describe the board in <perception></perception>. Then output your thinking process in <think></think> and final action in <answer>(row1, col1) (row2, col2)</answer>.
        """
        self.num_actions = GRID_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE
    
    def parse_action(self, response_text):
        """Parse the model output to extract the action coordinates."""
        action_match = re.search(r"<answer>\((\d+),\s*(\d+)\)\s*\((\d+),\s*(\d+)\)</answer>", response_text, flags=re.DOTALL)
        if action_match:
            # Convert to a single action index
            row1 = int(action_match.group(1))
            col1 = int(action_match.group(2))
            row2 = int(action_match.group(3))
            col2 = int(action_match.group(4))
            return row1*GRID_SIZE*GRID_SIZE*GRID_SIZE + col1*GRID_SIZE*GRID_SIZE + row2*GRID_SIZE + col2
        
        # Fallback: extract a single integer
        action_match = re.search(r"<answer>.*?(\d+).*?</answer>", response_text, flags=re.DOTALL)
        if action_match:
            a = int(action_match.group(1))
            return a*GRID_SIZE*GRID_SIZE*GRID_SIZE + a*GRID_SIZE*GRID_SIZE + a*GRID_SIZE + a
        else:
            print("No valid action found in the response.")
            return GRID_SIZE*GRID_SIZE*GRID_SIZE*GRID_SIZE - 1  # Default invalid action
    
    def gym_env_func(self):
        """Create and return a new environment instance."""
        return Match3GameEnv(render_mode="rgb_array")
    
    def reward_shaping(self, reward):
        """Shape the reward to provide better learning signals."""
        return 1 if reward > 0 else -1
    
    def state_to_text(self, state):
        """Convert the board state to a text representation."""
        text_board = []
        for row in state:
            row_text = []
            for cell in row:
                if cell:
                    color_name = TILE_COLORS_MATCH.get(cell[0], "Unknown")
                    shape_name = cell[1]
                    row_text.append(f"{color_name} {shape_name}")
                else:
                    row_text.append("Empty")
            text_board.append(" | ".join(row_text))
        return "\n".join(text_board)
    
    def get_text_observation(self, vec_env):
        """Get text observations from the vectorized environment."""
        states = vec_env.get_attr("board")
        text_observations = [self.state_to_text(state) for state in states]
        return text_observations


# For testing the game environment
if __name__ == "__main__":
    env = Match3GameEnv(render_mode="human")
    observation, info = env.reset()
    
    for _ in range(100):
        # Random valid action (adjacent cells)
        row1 = random.randint(0, GRID_SIZE-1)
        col1 = random.randint(0, GRID_SIZE-1)
        # Choose an adjacent cell
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction = random.choice(directions)
        row2 = row1 + direction[0]
        col2 = col1 + direction[1]
        
        # Ensure coordinates are within bounds
        if 0 <= row2 < GRID_SIZE and 0 <= col2 < GRID_SIZE:
            action = row1*GRID_SIZE*GRID_SIZE*GRID_SIZE + col1*GRID_SIZE*GRID_SIZE + row2*GRID_SIZE + col2
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game over! Final score: {info['score']}")
                observation, info = env.reset()
        
        time.sleep(0.5)  # Slow down the random play for visibility
    
    env.close()