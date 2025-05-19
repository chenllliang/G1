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

# 导入CIFAR-10相关库
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import io

from .game import VGameEnv

# ----------------- Constants Definition -----------------
# Match Game configuration parameters
GRID_SIZE = 6          # Board size is 6x6
TILE_SIZE = 80         # Each tile is 80 pixels in size
SCREEN_WIDTH = GRID_SIZE * TILE_SIZE
SCREEN_HEIGHT = GRID_SIZE * TILE_SIZE + 40  # Additional 40 pixels for score display
FPS = 60

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Semi-transparent yellow

# CIFAR-10 类别
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class CIFAR10TileProvider:
    """
    管理CIFAR-10图像的类，用于Match Game的瓷砖。
    每个类别将有多个可以使用的图像。
    """
    def __init__(self, cache_size=10):
        # 加载CIFAR-10数据集
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform
        )
        
        # 按类组织图像
        self.class_images = {i: [] for i in range(10)}
        for idx in range(len(self.dataset)):
            img, label = self.dataset[idx]
            self.class_images[label].append(img)
        
        # 渲染图像的缓存
        self.image_cache = {}
        self.cache_size = cache_size
        
    def get_surface(self, class_id, image_idx=None, size=(TILE_SIZE, TILE_SIZE)):
        """
        获取指定CIFAR-10类和可选图像索引的pygame surface。
        如果image_idx为None，则从该类中随机选择一个图像。
        """
        # 创建缓存键
        if image_idx is None:
            image_idx = random.randint(0, len(self.class_images[class_id]) - 1)
        
        cache_key = (class_id, image_idx, size)
        
        # 检查是否在缓存中
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # 获取图像张量
        img_tensor = self.class_images[class_id][image_idx]
        
        # 将张量转换为PIL图像
        img_pil = self.tensor_to_pil(img_tensor)
        
        # 调整图像大小
        img_pil = img_pil.resize(size, Image.LANCZOS)
        
        # 将PIL转换为Pygame surface
        surface = self.pil_to_surface(img_pil)
        
        # 缓存surface
        if len(self.image_cache) >= self.cache_size:
            # 如果缓存已满，则删除一个随机键
            random_key = random.choice(list(self.image_cache.keys()))
            del self.image_cache[random_key]
            
        self.image_cache[cache_key] = surface
        
        return surface
    
    @staticmethod
    def tensor_to_pil(tensor):
        """将张量转换为PIL图像。"""
        tensor = tensor.mul(255).byte()
        tensor = tensor.cpu().numpy().transpose((1, 2, 0))
        return Image.fromarray(tensor)
    
    @staticmethod
    def pil_to_surface(pil_image):
        """将PIL图像转换为Pygame surface。"""
        img_str = io.BytesIO()
        pil_image.save(img_str, format="PNG")
        img_str.seek(0)
        return pygame.image.load(img_str)

# 创建CIFAR10TileProvider的全局实例
cifar10_provider = None


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
        """
        初始化游戏状态，包括为每个类别预先选择不同的图像
        """
        global cifar10_provider
        if cifar10_provider is None:
            cifar10_provider = CIFAR10TileProvider()
        
        # 1. 选择类别
        needed_unique = (GRID_SIZE * GRID_SIZE) // 4
        selected_classes = random.sample(range(10), min(10, needed_unique))
        
        # 如果需要的类型比可用的CIFAR-10类别多，则重复使用一些类别
        if len(selected_classes) < needed_unique:
            additional_needed = needed_unique - len(selected_classes)
            additional_classes = random.choices(selected_classes, k=additional_needed)
            selected_classes.extend(additional_classes)
        
        # 2. 为每个类别选择4个不同的图像索引
        self.class_image_indices = {}
        for class_id in selected_classes:
            class_images = cifar10_provider.class_images[class_id]
            image_indices = random.sample(
                range(len(class_images)),
                min(4, len(class_images))  # 确保不要超出该类别的图像数量
            )
            
            # 如果该类别的图像不足4个，则重复使用一些图像
            if len(image_indices) < 4:
                additional_needed = 4 - len(image_indices)
                additional_indices = random.choices(image_indices, k=additional_needed)
                image_indices.extend(additional_indices)
            
            self.class_image_indices[class_id] = image_indices
        
        # 存储类别列表作为tile_types
        self.tile_types = list(self.class_image_indices.keys())
        
        # 创建棋盘并初始化其他游戏状态
        self.board = self.create_board()
        self.selected_tile = None
        self.game_over = False
        self.score = 0
        self.remaining_tiles = GRID_SIZE * GRID_SIZE
        self.step_count = 0
        self.close_render = False

    def set_close_render(self, close_render):
        """
        设置是否关闭渲染
        """
        # print(f"Set close_render: {close_render}")
        # print(f"close_render: {self.close_render}")
        self.close_render = close_render

    def create_board(self):
        """
        使用预先选择的类别和图像索引创建棋盘
        """
        board = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        # 准备瓦片对
        tile_pairs = []
        for class_id, image_indices in self.class_image_indices.items():
            # 将类别与图像索引配对后添加到瓦片对列表
            for img_idx in image_indices:
                tile_pairs.append((class_id, img_idx))
        
        # 打乱瓦片对
        random.shuffle(tile_pairs)
        
        # 放置瓦片到棋盘上
        index = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                board[i][j] = tile_pairs[index]
                index += 1
                
        return board

    def draw_board(self):
        """
        Draw the entire board, all tiles, and the score information on the screen.
        使用CIFAR-10图像绘制瓦片。
        """
        global cifar10_provider
        if cifar10_provider is None:
            cifar10_provider = CIFAR10TileProvider()
        
        self.screen.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
                if self.board[i][j] is not None:
                    class_id, img_idx = self.board[i][j]
                    tile_surface = cifar10_provider.get_surface(class_id, img_idx)
                    self.screen.blit(tile_surface, rect)
        
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
          - Both tiles must exist and have the same class ID (even if they have different image indices).
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
        
        # 只比较类别ID而不是完整的元组
        class_id1 = self.board[row1][col1][0]
        class_id2 = self.board[row2][col2][0]
        if class_id1 != class_id2:
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
                        if (self.board[i1][j1][0] == self.board[i2][j2][0] and
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
        obs = np.zeros((self.observation_space.shape[0], self.observation_space.shape[1], 3), dtype=np.uint8)  # 全黑图像
        if not self.close_render:
            # print(f"Step {self.close_render}")
            obs = self.render(mode="rgb_array")
        # 如果全部消除完了，就reset
        if done:
            obs, info = self.reset()
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
        """
        清理 pygame 资源
        """
        try:
            # 在调用 pygame.quit() 之前检查 pygame 是否已初始化
            if pygame.get_init():
                pygame.quit()
        except Exception as e:
            print(f"Error during close: {e}")


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


class GameMatchCifar(VGameEnv):
    def __init__(self):
        super().__init__()
        self.game_name = "Match Game with CIFAR-10"
        self.game_prompt = "You are playing a Shisen-sho puzzle game that uses CIFAR-10 images. Each tile on the board corresponds to one of the CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The objective is to find a pair of tiles that belong to the same class and can be connected with a path that does not cross any other tiles and makes at most two turns.\nPlease analyze the game board and identify two matching tiles that can be connected according to these rules.\nReturn your answer as follows:\n1. First coordinate: (row1, col1)\n2. Second coordinate: (row2, col2)\nWhere row and col are 0-indexed numbers such as (0, 1), starting from the top-left of the board.\nFirst describe the board in <perception></perception>. Then output your thinking process in <think></think> and final action in <answer>(row1, col1) (row2, col2)</answer>."
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
        return reward if reward > 0 else 0

    def state_to_text(self, state):
        """
        将状态转换为文本观察。每个瓷砖都用其CIFAR-10类别表示。
        """
        text_rows = []
        for row in state:
            text_cells = []
            for cell in row:
                if cell is None:
                    text_cells.append(" - ")
                else:
                    class_id, _ = cell
                    text_cells.append(f" {CIFAR10_CLASSES[class_id]} ")
            text_rows.append("|".join(text_cells))
        return "\n".join(text_rows)

    def get_text_observation(self, vec_env):
        """
        Get the text observation from the vectorized environment. The 
        """
        
        states = [ i for i in vec_env.get_attr("board")]

        text_observations = [self.state_to_text(state) for state in states]

        return text_observations