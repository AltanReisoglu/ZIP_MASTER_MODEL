"""
ZIP Game Environment - OpenEnv Compatible

TRL openenv ve OpenSpiel tarzında tasarlanmış RL ortamı.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
import numpy as np



class ZipAction(BaseModel):
    """ZIP oyunu aksiyonu"""
    direction: str  # "up", "down", "left", "right"
    
    def to_dict(self) -> dict:
        return {"direction": self.direction}
    
    @classmethod
    def from_dict(cls, d: dict) -> "ZipAction":
        return cls(direction=d["direction"])



class ZipObservation(BaseModel):
    
    board: List[List[int]]
    current_pos: Tuple[int, int]
    current_target: int
    visited: List[Tuple[int, int]]
    legal_actions: List[str]
    info_state: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "current_pos": self.current_pos,
            "current_target": self.current_target,
            "visited": self.visited,
            "legal_actions": self.legal_actions,
            "info_state": self.info_state,
        }
    
    def to_text(self) -> str:
        """Turn into text format"""
        size = len(self.board)
        lines = [f"Target: {self.current_target} | Pos: {self.current_pos}"]
        for r in range(size):
            row = ""
            for c in range(size):
                if (r, c) == self.current_pos:
                    row += "[*]"
                elif (r, c) in self.visited:
                    row += " · "
                elif self.board[r][c] > 0:
                    row += f" {self.board[r][c]} "
                else:
                    row += "   "
            lines.append(row)
        return "\n".join(lines)


class ZipResult(BaseModel):
    """Step result - OpenEnv compatible"""
    observation: ZipObservation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ZipEnv:
    """
    ZIP RL Environment - OpenEnv/OpenSpiel Compatible
    
    Usage:
        env = ZipEnv(size=6, num_count=10)
        result = env.reset()
        result = env.step(ZipAction("right"))
    """
    
    ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    
    def __init__(self, size: int = 6, num_count: int = 10 ,seed: int = None):
        self.size = size
        self.num_count = num_count
        self.rng = np.random.RandomState(seed)
        self._reset_state()

    
    def _reset_state(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.number_pos = {}
        self.current_pos = (0, 0)
        self.current_target = 2
        self.visited = set()
        self.done = False
    
    def reset(self, seed: int = None) -> ZipResult:
        """Yeni oyun başlat"""
        if seed:
            self.rng = np.random.RandomState(seed)
        
        self._reset_state()
        
        # Rastgele board oluştur
        positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(positions)
        
        for i in range(self.num_count):
            r, c = positions[i]
            self.board[r, c] = i + 1
            self.number_pos[i + 1] = (r, c)
        
        self.current_pos = self.number_pos[1]
        self.visited = {self.current_pos}
        
        return ZipResult(
            observation=self._get_obs(),
            reward=0.0,
            done=False,
            info={"message": "Game started"}
        )
    
    def step(self, action: ZipAction) -> ZipResult:
        """Aksiyon uygula"""
        if self.done:
            return ZipResult(self._get_obs(), 0.0, True, {"error": "game_over"})
        
        direction = action.direction if isinstance(action, ZipAction) else action
        
        if direction not in self.ACTIONS:
            return ZipResult(self._get_obs(), -0.1, False, {"error": "invalid"})
        
        dr, dc = self.ACTIONS[direction]
        nr, nc = self.current_pos[0] + dr, self.current_pos[1] + dc
        
        # Sınır kontrolü
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return ZipResult(self._get_obs(), -0.1, False, {"error": "bounds"})
        
        # Ziyaret kontrolü
        if (nr, nc) in self.visited:
            return ZipResult(self._get_obs(), -0.1, False, {"error": "visited"})
        
        # Hareket
        self.current_pos = (nr, nc)
        self.visited.add(self.current_pos)
        
        cell = self.board[nr, nc]
        reward = 0.01
        info = {"cell": int(cell)}
        
        if cell == self.current_target:
            reward = 1.0
            self.current_target += 1
            if self.current_target > self.num_count:
                reward = 10.0 * (len(self.visited) / (self.size ** 2))
                self.done = True
                info["win"] = True
        
        if not self._legal_actions() and not self.done:
            reward = -1.0
            self.done = True
            info["stuck"] = True
        
        return ZipResult(self._get_obs(), reward, self.done, info)
    
    def _legal_actions(self) -> List[str]:
        """Geçerli aksiyonlar"""
        legal = []
        r, c = self.current_pos
        for name, (dr, dc) in self.ACTIONS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if (nr, nc) not in self.visited:
                    legal.append(name)
        return legal
    
    def _get_obs(self) -> ZipObservation:
        """Gözlem oluştur"""
        # info_state: board'u flatten et (OpenSpiel uyumu için)
        info_state = self.board.flatten().astype(float).tolist()
        
        return ZipObservation(
            board=self.board.tolist(),
            current_pos=self.current_pos,
            current_target=self.current_target,
            visited=list(self.visited),
            legal_actions=self._legal_actions(),
            info_state=info_state,
        )
    
    def render(self) -> str:
        return self._get_obs().to_text()


# Test
if __name__ == "__main__":
    import random
    
    env = ZipEnv(size=6, num_count=8, seed=42)
    result = env.reset()
    print(result.observation.to_text())
    
    for _ in range(20):
        legal = result.observation.legal_actions
        if not legal:
            break
        action = ZipAction(random.choice(legal))
        result = env.step(action)
        print(f"\n{action.direction} -> r:{result.reward:.2f}")
        print(result.observation.to_text())
        if result.done:
            break
