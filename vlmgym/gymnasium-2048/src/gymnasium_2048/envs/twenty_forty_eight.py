from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
WINDOW_SCORE_HEIGHT = 60
WINDOW_BG_COLOR = (250, 248, 238)

BOARD_PADDING = 20
BOARD_BG_COLOR = (186, 172, 160)
TILE_PADDING = 5
TILE_COLOR_MAP = {
    0: (204, 193, 178),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TILE_COLOR_DEFAULT = (60, 58, 50)
BORDER_RADIUS = 4

FONT_NAME = "Comic Sans MS"
FONT_DARK_COLOR = (119, 110, 101)
FONT_LIGHT_COLOR = (249, 246, 242)
FONT_SCORE_COLOR = (0, 0, 0)
FONT_SIZE = 40


class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 4,
        max_pow: int = 16,
    ) -> None:
        assert size >= 2, "size must be greater of equal than 2"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT, WINDOW_WIDTH, 3),
            dtype=np.uint8,
        )

        # 0: up, 1: right, 2: down, 3: left, 4: still
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None
        
        self.size = size
        self.max_pow = max_pow

    def _get_obs(self) -> ObsType:
        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT))
        canvas.fill(WINDOW_BG_COLOR)

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        self._draw_board(canvas=canvas)
        self._draw_score(canvas=canvas)

        rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        
        return rgb_array

    def _get_info(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "step_score": self.step_score,
            "total_score": self.total_score,
            "max": np.max(self.board),
            "is_legal": self.is_legal,
            "illegal_count": self.illegal_count,
        }

    def _spawn_tile(self) -> None:
        rows, cols = np.where(self.board == 0)
        index = self.np_random.choice(len(rows))
        value = 1 if self.np_random.random() > 0.1 else 2
        self.board[rows[index], cols[index]] = value

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.board = np.zeros(
            (self.size, self.size),
            dtype=np.uint8,
        )
        self.step_score = 0
        self.total_score = 0
        self.is_legal = True
        self.illegal_count = 0

        self._spawn_tile()
        self._spawn_tile()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    @staticmethod
    def _transpose(board: np.ndarray) -> np.ndarray:
        """Transpose a matrix."""
        return np.transpose(board)

    @staticmethod
    def _reverse(board: np.ndarray) -> np.ndarray:
        """Reverse a matrix."""
        return np.flipud(board)

    @staticmethod
    def _cover_up(board: np.ndarray) -> np.ndarray:
        """Cover the most antecedent zeros with non-zero number."""
        cover_board = np.zeros_like(board)

        for col in range(board.shape[1]):
            up = 0
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    cover_board[up, col] = board[row, col]
                    up += 1

        return cover_board

    @staticmethod
    def _merge(board: np.ndarray) -> tuple[np.ndarray, int]:
        """Verify if a merge is possible and execute."""
        score = 0

        for row in range(1, board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] != 0 and board[row, col] == board[row - 1, col]:
                    score += 2 ** (board[row, col] + 1)
                    board[row - 1, col] = board[row - 1, col] + 1
                    board[row, col] = 0

        return board, score

    @classmethod
    def _up(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._cover_up(board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        return next_board, score

    @classmethod
    def _right(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(cls._transpose(board))
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(cls._reverse(next_board))
        return next_board, score

    @classmethod
    def _down(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._reverse(next_board)
        return next_board, score

    @classmethod
    def _left(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._transpose(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(next_board)
        return next_board, score

    @classmethod
    def _still(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        """Return board unchanged with zero score."""
        return board.copy(), 0

    @classmethod
    def apply_action(
        cls,
        board: np.ndarray,
        action: ActType,
    ) -> tuple[np.ndarray, int, bool]:
        """Apply an action to the board without spawning a new tile."""
        action_func = (cls._up, cls._right, cls._down, cls._left, cls._still)
        next_board, score = action_func[action](board)
        is_legal = not np.array_equal(board, next_board) if action != 4 else True
        return next_board, score, is_legal

    @staticmethod
    def is_terminated(board: np.ndarray) -> bool:
        """Check if the game is terminated or not."""
        # Verify zero entries
        if (board == 0).any():
            return False

        # Verify possible merges
        for row in range(1, board.shape[0]):
            for col in range(1, board.shape[1]):
                if (
                    board[row, col] == board[row, col - 1]
                    or board[row, col] == board[row - 1, col]
                ):
                    return False

        # Verify possible merges in first column
        for row in range(1, board.shape[0]):
            if board[row, 0] == board[row - 1, 0]:
                return False

        # Verify possible merges in first row
        for col in range(1, board.shape[1]):
            if board[0, col] == board[0, col - 1]:
                return False

        return True

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        next_board, self.step_score, self.is_legal = self.apply_action(
            board=self.board,
            action=action,
        )
        self.total_score += self.step_score
        if self.is_legal and action != 4:  # Don't spawn new tile for "still" action
            self.board = next_board
            self._spawn_tile()
        elif not self.is_legal:
            self.illegal_count += 1
        else:  # For "still" action when legal
            self.board = next_board

        observation = self._get_obs()
        reward = -1.0 if action == 4 else self.step_score  # Apply -1.0 reward for "still" action
        terminated = self.is_terminated(board=self.board)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_value(self, row: int, col: int) -> int:
        return 2 ** self.board[row, col] if self.board[row, col] > 0 else 0

    @staticmethod
    def _get_background_color(value: int) -> tuple[int, int, int]:
        return TILE_COLOR_MAP.get(value, TILE_COLOR_DEFAULT)

    @staticmethod
    def _get_text_color(value: int) -> tuple[int, int, int]:
        return FONT_DARK_COLOR if value < 8 else FONT_LIGHT_COLOR

    def _draw_board(self, canvas: pygame.Surface) -> None:
        board_left = BOARD_PADDING
        board_right = BOARD_PADDING
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        board_height = WINDOW_HEIGHT - 2 * BOARD_PADDING
        tile_width = (board_width - 2 * TILE_PADDING) // self.board.shape[1]
        tile_height = (board_height - 2 * TILE_PADDING) // self.board.shape[0]
        pygame.draw.rect(
            surface=canvas,
            color=BOARD_BG_COLOR,
            rect=(board_left, board_right, board_width, board_height),
            border_radius=BORDER_RADIUS,
        )
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                value = self._get_value(row=row, col=col)
                rect = pygame.Rect(
                    board_left + col * tile_width + 2 * TILE_PADDING,
                    board_right + row * tile_height + 2 * TILE_PADDING,
                    tile_width - 2 * TILE_PADDING,
                    tile_height - 2 * TILE_PADDING,
                )
                pygame.draw.rect(
                    surface=canvas,
                    color=self._get_background_color(value=value),
                    rect=rect,
                    border_radius=BORDER_RADIUS,
                )
                if value == 0:
                    continue
                text_surface = self.font.render(
                    str(value),
                    True,
                    self._get_text_color(value=value),
                )
                text_rect = text_surface.get_rect(center=rect.center)
                canvas.blit(source=text_surface, dest=text_rect)

    def _draw_score(self, canvas: pygame.Surface) -> None:
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        score_surface = self.font.render(
            f"Score: {self.total_score}",
            True,
            FONT_SCORE_COLOR,
        )
        score_height = self.font.get_height()
        score_rect = pygame.Rect(
            BOARD_PADDING,
            WINDOW_HEIGHT + (WINDOW_SCORE_HEIGHT - score_height) // 2,
            board_width,
            score_height,
        )
        canvas.blit(source=score_surface, dest=score_rect)

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT)
            )
            pygame.display.set_caption("2048")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT))
        canvas.fill(WINDOW_BG_COLOR)

        self._draw_board(canvas=canvas)
        self._draw_score(canvas=canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
