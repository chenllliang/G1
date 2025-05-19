import base64
import json
import os
import random
import re
import time
from collections import deque
from multiprocessing import Pool

import gymnasium as gym
import imageio
import numpy as np
import pygame
import ale_py
from openai import OpenAI
from tqdm import tqdm
from .game import VGameEnv

# ----------------- Constants Definition -----------------
# Match Game configuration parameters
GRID_SIZE = 12          # Board size is 6x6
TILE_SIZE = 80         # Each tile is 80 pixels in size
SCREEN_WIDTH = GRID_SIZE * TILE_SIZE
SCREEN_HEIGHT = GRID_SIZE * TILE_SIZE + 40  # Additional 40 pixels for score display
FPS = 60

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Semi-transparent yellow

# Tile colors and shapes
TILE_COLORS = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    # (128, 0, 0),     # Maroon
    # (0, 128, 0),     # Dark green
    # (0, 0, 128),     # Navy
    # (128, 128, 0),   # Olive
    # (128, 0, 128),   # Purple
    # (0, 128, 128),   # Teal
    # (255, 165, 0),   # Orange
    # (139, 69, 19),   # Brown
    # (70, 130, 180),  # Steel blue
    # (219, 112, 147)  # Pink
]
SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"]

TILE_COLORS_MATCH = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 0, 255): "Magenta",
    (0, 255, 255): "Cyan",
    (128, 0, 0): "Maroon",
    (0, 128, 0): "Dark green",
    (0, 0, 128): "Navy",
    (128, 128, 0): "Olive",
    (128, 0, 128): "Purple",
    (0, 128, 128): "Teal",
    (255, 165, 0): "Orange",
    (139, 69, 19): "Brown",
    (70, 130, 180): "Steel blue",
    (219, 112, 147): "Pink"
}


# ----------------- Match Game Gym Environment -----------------
class MatchGameEnv(gym.Env):
    """
    Gymnasium environment encapsulation for the Match Game.
    Observation: RGB image of the screen.
    Action: (row1, col1, row2, col2) with each coordinate in the discrete range [0, GRID_SIZE-1].
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode='rgb_array'):
        super().__init__()
        self.render_mode = render_mode
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        # Define action_space as a 4-dimensional discrete space
        self.action_space = gym.spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE])
        # Observation: RGB array of the screen image with shape (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        self._init_game()

    def _init_game(self):
        self.tile_types = self.create_tile_types()
        self.board = self.create_board()
        self.selected_tile = None
        self.game_over = False
        self.score = 0
        self.remaining_tiles = GRID_SIZE * GRID_SIZE
        self.step_count = 0

    def create_tile_types(self):
        """
        Create tile types using colors and shapes.
        Each tile type will appear 4 times on the board.
        """
        needed_unique = (GRID_SIZE * GRID_SIZE) // 4
        possible_types = [(color, shape) for color in TILE_COLORS for shape in SHAPES]
        random.shuffle(possible_types)
        tile_types = possible_types[:needed_unique]
        if len(tile_types) < needed_unique:
            while len(tile_types) < needed_unique:
                tile_types.append(random.choice(tile_types))
        return tile_types

    def create_board(self):
        """
        Generate the board (2D list) and randomly distribute all tiles.
        """
        board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        tile_pairs = []
        needed_unique = (GRID_SIZE * GRID_SIZE) // 4
        available_types = self.tile_types[:needed_unique]
        for tile_type in available_types:
            tile_pairs.extend([tile_type] * 4)
        random.shuffle(tile_pairs)
        index = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                board[i][j] = tile_pairs[index]
                index += 1
        return board

    def draw_shape(self, surface, shape, color, rect):
        """
        Draw the specified tile (with its shape and color) within the given rectangular area.
        """
        x, y, w, h = rect
        padding = 10
        inner_rect = (x + padding, y + padding, w - 2 * padding, h - 2 * padding)
        if shape == "circle":
            pygame.draw.ellipse(surface, color, inner_rect)
        elif shape == "square":
            pygame.draw.rect(surface, color, inner_rect)
        elif shape == "triangle":
            points = [
                (x + w // 2, y + padding),
                (x + padding, y + h - padding),
                (x + w - padding, y + h - padding)
            ]
            pygame.draw.polygon(surface, color, points)
        elif shape == "diamond":
            points = [
                (x + w // 2, y + padding),
                (x + w - padding, y + h // 2),
                (x + w // 2, y + h - padding),
                (x + padding, y + h // 2)
            ]
            pygame.draw.polygon(surface, color, points)
        elif shape == "cross":
            thickness = 8
            pygame.draw.rect(surface, color, (x + padding, y + h // 2 - thickness // 2, w - 2 * padding, thickness))
            pygame.draw.rect(surface, color, (x + w // 2 - thickness // 2, y + padding, thickness, h - 2 * padding))
        elif shape == "star":
            thickness = 8
            pygame.draw.line(surface, color, (x + padding, y + padding), (x + w - padding, y + h - padding), thickness)
            pygame.draw.line(surface, color, (x + padding, y + h - padding), (x + w - padding, y + padding), thickness)
        # Draw black border
        if shape == "circle":
            pygame.draw.ellipse(surface, BLACK, inner_rect, 2)
        elif shape == "square":
            pygame.draw.rect(surface, BLACK, inner_rect, 2)
        elif shape in ["triangle", "diamond"]:
            if shape == "triangle":
                points = [
                    (x + w // 2, y + padding),
                    (x + padding, y + h - padding),
                    (x + w - padding, y + h - padding)
                ]
            else:
                points = [
                    (x + w // 2, y + padding),
                    (x + w - padding, y + h // 2),
                    (x + w // 2, y + h - padding),
                    (x + padding, y + h // 2)
                ]
            pygame.draw.polygon(surface, BLACK, points, 2)

    def draw_board(self):
        """
        Draw the entire board, all tiles, and the score information on the screen.
        """
        self.screen.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
                if self.board[i][j] is not None:
                    color, shape = self.board[i][j]
                    self.draw_shape(self.screen, shape, color, rect)
        if self.selected_tile:
            i, j = self.selected_tile
            highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            highlight.fill(HIGHLIGHT_COLOR)
            self.screen.blit(highlight, (j * TILE_SIZE, i * TILE_SIZE))
        # Draw score and remaining tile count
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}   Remaining: {self.remaining_tiles}", True, BLACK)
        self.screen.blit(score_text, (10, SCREEN_HEIGHT - 35))
        if self.render_mode == "human":
            pygame.display.flip()

    def check_straight_line(self, pos1, pos2):
        """
        Check if two tiles can be connected by a straight line.
        """
        row1, col1 = pos1
        row2, col2 = pos2
        if row1 == row2:
            start_col, end_col = min(col1, col2), max(col1, col2)
            for col in range(start_col + 1, end_col):
                if self.board[row1][col] is not None:
                    return False
            return True
        if col1 == col2:
            start_row, end_row = min(row1, row2), max(row1, row2)
            for row in range(start_row + 1, end_row):
                if self.board[row][col1] is not None:
                    return False
            return True
        return False

    def check_one_turn(self, pos1, pos2):
        """
        Check if two tiles can be connected with one turn (L-shape).
        """
        row1, col1 = pos1
        row2, col2 = pos2
        corners = [(row1, col2), (row2, col1)]
        for cr, cc in corners:
            if self.board[cr][cc] is not None:
                continue
            if self.check_straight_line(pos1, (cr, cc)) and self.check_straight_line((cr, cc), pos2):
                return True
        return False

    def check_two_turns(self, pos1, pos2):
        """
        Check if two tiles can be connected with two turns (Z-shape).
        """
        row1, col1 = pos1
        row2, col2 = pos2
        # Horizontal-vertical-horizontal
        for row in range(GRID_SIZE):
            if row == row1 or row == row2:
                continue
            corner1 = (row, col1)
            corner2 = (row, col2)
            if (self.board[corner1[0]][corner1[1]] is None and
                self.board[corner2[0]][corner2[1]] is None and
                self.check_straight_line(pos1, corner1) and
                self.check_straight_line(corner1, corner2) and
                self.check_straight_line(corner2, pos2)):
                return True
        # Vertical-horizontal-vertical
        for col in range(GRID_SIZE):
            if col == col1 or col == col2:
                continue
            corner1 = (row1, col)
            corner2 = (row2, col)
            if (self.board[corner1[0]][corner1[1]] is None and
                self.board[corner2[0]][corner2[1]] is None and
                self.check_straight_line(pos1, corner1) and
                self.check_straight_line(corner1, corner2) and
                self.check_straight_line(corner2, pos2)):
                return True
        return False

    def can_connect_with_border(self, pos1, pos2):
        """
        Use an expanded board (a border of empty space) to check if tiles can be connected.
        Allows the path to go through the border area.
        """
        R = GRID_SIZE + 2
        C = GRID_SIZE + 2
        start = (pos1[0] + 1, pos1[1] + 1)
        end = (pos2[0] + 1, pos2[1] + 1)
        allowed_turns = 2

        def is_empty(cell):
            x, y = cell
            if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
                if cell == start or cell == end:
                    return True
                return self.board[x - 1][y - 1] is None
            return True

        dq = deque()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = {}
        for d, (dx, dy) in enumerate(directions):
            nx, ny = start[0] + dx, start[1] + dy
            if 0 <= nx < R and 0 <= ny < C and is_empty((nx, ny)):
                dq.append((nx, ny, d, 0))
                visited[(nx, ny, d)] = 0

        while dq:
            x, y, direction, turns = dq.popleft()
            if (x, y) == end:
                return True
            if turns > allowed_turns:
                continue
            for new_dir, (dx, dy) in enumerate(directions):
                nx, ny = x + dx, y + dy
                new_turns = turns + (0 if new_dir == direction else 1)
                if new_turns > allowed_turns:
                    continue
                if 0 <= nx < R and 0 <= ny < C and is_empty((nx, ny)):
                    state = (nx, ny, new_dir)
                    if state in visited and visited[state] <= new_turns:
                        continue
                    visited[state] = new_turns
                    dq.append((nx, ny, new_dir, new_turns))
        return False

    def can_connect(self, pos1, pos2):
        """
        Check if two tiles can be connected according to the rules:
          - Both tiles must exist and be identical.
          - They can be connected via a straight line, one turn, or two turns,
            or by extending the path outside the board borders.
        """
        row1, col1 = pos1
        row2, col2 = pos2
        if not (0 <= row1 < GRID_SIZE and 0 <= col1 < GRID_SIZE and
                0 <= row2 < GRID_SIZE and 0 <= col2 < GRID_SIZE):
            return False
        if self.board[row1][col1] is None or self.board[row2][col2] is None:
            return False
        if self.board[row1][col1] != self.board[row2][col2]:
            return False
        if (row1, col1) == (row2, col2):
            return False
        if (self.check_straight_line(pos1, pos2) or
            self.check_one_turn(pos1, pos2) or
            self.check_two_turns(pos1, pos2)):
            return True
        if self.can_connect_with_border(pos1, pos2):
            return True
        return False

    def check_game_over(self):
        """
        Check if there are any possible matching pairs left on the board.
        """
        for i1 in range(GRID_SIZE):
            for j1 in range(GRID_SIZE):
                if self.board[i1][j1] is None:
                    continue
                for i2 in range(GRID_SIZE):
                    for j2 in range(GRID_SIZE):
                        if (i1 == i2 and j1 == j2) or self.board[i2][j2] is None:
                            continue
                        if (self.board[i1][j1] == self.board[i2][j2] and
                            self.can_connect((i1, j1), (i2, j2))):
                            return False
        return True

    def match_tiles(self, pos1, pos2):
        """
        If the two tiles can be connected (matched), remove them from the board and update the score.
        """
        if self.can_connect(pos1, pos2):
            self.board[pos1[0]][pos1[1]] = None
            self.board[pos2[0]][pos2[1]] = None
            self.score += 1
            self.remaining_tiles -= 2
            if self.remaining_tiles == 0:
                self.game_over = True
            elif self.check_game_over():
                self.game_over = True
            return True
        return False

    def step(self, action):
        """
        Execute one move based on the action = (row1, col1, row2, col2).
        If the tiles are matched, gain 1 points; otherwise, lose 1 point.
        Returns (observation, reward, done, truncated, info).
        """
        if self.game_over:
            return self.render(mode="rgb_array"), 0, True, False, {"score": self.score, "remaining_tiles": self.remaining_tiles}
        # row1, col1, row2, col2 = action
        row1 = action // (6*6*6)
        col1 = (action % (6*6*6)) // (6*6)
        row2 = (action % (6*6)) // 6
        col2 = action % 6
        pos1 = (row1, col1)
        pos2 = (row2, col2)
        if action==36*36+1:
            reward = -1
        elif self.can_connect(pos1, pos2):
            success = self.match_tiles(pos1, pos2)
            reward = 1 if success else -1
        else:
            reward = -1
        self.step_count += 1
        done = self.game_over or self.check_game_over()
        info = {"score": self.score, "remaining_tiles": self.remaining_tiles}
        obs = self.render(mode="rgb_array")
        # 如果全部消除完了，就reset
        if done:
            self.reset()
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self._init_game()
        return self.render(mode="rgb_array"), {}

    def render(self, mode="rgb_array"):
        self.draw_board()
        if mode == "human":
            return None
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        return None

    def close(self):
        pygame.quit()


# ----------------- Sandbox Code -----------------

def call_random_action(image_path_lists, prompt_lists):
    """
    Generate random actions for the Match Game environment.
    Returns a list of response texts.
    """
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


class GameMatch(VGameEnv):
    def __init__(self):
        super().__init__()
        self.game_name = "Match Game"
        self.game_prompt = "You are playing a 'Shisen-sho' puzzle game.\nThe goal is to match pairs of identical tiles by connecting them with a path that has at most 2 turns and doesn't cross any other tiles.\nThe tiles are distinguished by their color and shape:\n- Red, Green, Blue, Yellow, Magenta, Cyan, etc.\n- Shapes include: circle, square, triangle, diamond, cross, star, etc.\nPlease analyze the game board and identify two matching tiles that can be connected according to these rules.\nReturn your answer as follows:\n1. First coordinate: (row1, col1)\n2. Second coordinate: (row2, col2)\nWhere row and col are 0-indexed numbers such as (0, 1), starting from the top-left of the board.\nFirst describe the board in <perception></perception>. Then output your thinking process in <think></think> and final action in <answer>(row1, col1) (row2, col2)</answer>."
        self.num_actions = 36*36+1

    def parse_action(self,response_text):
        """
        Parse the model output.
        First, match the pattern <answer>(row1, col1) (row2, col2)</answer>.
        If no match is found, attempt to extract a single integer and construct a simple action (not recommended).
        """
        action_match = re.search(r"<answer>\((\d+),\s*(\d+)\)\s*\((\d+),\s*(\d+)\)</answer>", response_text, flags=re.DOTALL)
        if action_match:
            # return (int(action_match.group(1)),
            #         int(action_match.group(2)),
            #         int(action_match.group(3)),
            #         int(action_match.group(4)))
            # 转成四位数数字
            return int(action_match.group(1))*6*6*6+int(action_match.group(2))*6*6+int(action_match.group(3))*6+int(action_match.group(4))
        # Fallback: extract a single integer
        action_match = re.search(r"<answer>.*?(\d+).*?</answer>", response_text, flags=re.DOTALL)
        if action_match:
            a = int(action_match.group(1))
            # return (a, a, a, a)
            return a*6*6*6+a*6*6+a*6+a
        else:
            print("No valid action found in the response.")
            return 36*36+1

    def gym_env_func(self):
        return MatchGameEnv(render_mode="rgb_array")

    def reward_shaping(self, reward):
        """
        0->0, 8->3, 16->4, 32->5, 64->6, 128->7, 256->8, 512->9, 1024->10, 2048->11, 4096->12, 8192->13, 16384->14, 32768->15, 65536->16
        """
        return reward 

    def state_to_text(self, state):
        """
        Convert the state to a text observation. Each tile is represented by its color and shape.
        """
        return "\n".join(["|".join([" "+str(TILE_COLORS_MATCH[cell[0]]+" "+str(cell[1]))+" " if cell else " - " for cell in row]) for row in state])




    def get_text_observation(self, vec_env):
        """
        Get the text observation from the vectorized environment. The 
        """
        
        states = [ i for i in vec_env.get_attr("board")]

        text_observations = [self.state_to_text(state) for state in states]

        return text_observations