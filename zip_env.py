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
    direction: str  
    
    def __init__(self, direction: str = None, **data):
        if direction is not None and 'direction' not in data:
            data['direction'] = direction
        super().__init__(**data)
    
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
        """Turn into text format with grid display."""
        size = len(self.board)
        lines = [f"Target: {self.current_target} | Pos: {self.current_pos}"]
        
        # Top border
        lines.append("+" + "----+" * size)
        
        for r in range(size):
            row = "|"
            for c in range(size):
                if (r, c) == self.current_pos:
                    row += " *  |"
                elif (r, c) in self.visited:
                    row += " ·  |"
                elif self.board[r][c] > 0:
                    row += f" {self.board[r][c]:2} |"
                else:
                    row += "    |"
            lines.append(row)
            # Row separator
            lines.append("+" + "----+" * size)
        
        return "\n".join(lines)


class ZipResult(BaseModel):
    """Step result - OpenEnv compatible"""
    observation: ZipObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}
    
    def __init__(self, observation=None, reward=None, done=None, info=None, **data):
        if observation is not None and 'observation' not in data:
            data['observation'] = observation
        if reward is not None and 'reward' not in data:
            data['reward'] = reward
        if done is not None and 'done' not in data:
            data['done'] = done
        if info is not None and 'info' not in data:
            data['info'] = info
        super().__init__(**data)


class ZipEnv:
    """
    ZIP RL Environment - OpenEnv/OpenSpiel Compatible
    
    Usage:
        env = ZipEnv(size=6, num_count=10)
        result = env.reset()
        result = env.step(ZipAction("right"))
    """
    
    ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    
    def __init__(self, size: int = 6, num_count: int = 10, seed: int = None):
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
        total_cells = self.size ** 2
        coverage = len(self.visited) / total_cells
        
        reward = 0.01 + (0.05 * coverage)  # Small reward for each move + coverage bonus
        info = {"cell": int(cell), "coverage": coverage}
        
        if cell == self.current_target:
            reward = 1.0
            self.current_target += 1
            
            # Win condition: Reached all numbers AND visited all cells
            if self.current_target > self.num_count:
                if len(self.visited) == total_cells:
                    reward = 10.0  # Full win!
                    info["win"] = True
                    info["full_coverage"] = True
                else:
                    reward = 5.0 * coverage  # Partial win (numbers done but not all cells)
                    info["partial_win"] = True
                self.done = True
        
        if not self._legal_actions() and not self.done:
            reward = -1.0 + coverage  # Less penalty if high coverage
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
    
    def _generate_solvable_path(self, length: int) -> list[tuple[int, int]] | None:
        """
        Generate a Hamiltonian path that visits ALL cells.
        Uses Warnsdorff's heuristic: prefer moves to cells with fewer remaining neighbors.
        """
        total_cells = self.size * self.size
        
        # If length is less than total cells, use simpler method
        if length < total_cells:
            return self._generate_partial_path(length)
        
        max_attempts = 50
        
        for _ in range(max_attempts):
            # Try different starting positions
            start_r = self.rng.randint(0, self.size)
            start_c = self.rng.randint(0, self.size)
            
            path = self._try_hamiltonian_from(start_r, start_c)
            if path and len(path) == total_cells:
                return path
        
        return None
    
    def _count_unvisited_neighbors(self, r: int, c: int, visited: set) -> int:
        """Count how many unvisited neighbors a cell has."""
        count = 0
        for dr, dc in self.ACTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if (nr, nc) not in visited:
                    count += 1
        return count
    
    def _try_hamiltonian_from(self, start_r: int, start_c: int) -> list[tuple[int, int]] | None:
        """Try to find Hamiltonian path starting from given position."""
        total_cells = self.size * self.size
        path = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        
        while len(path) < total_cells:
            r, c = path[-1]
            
            # Get unvisited neighbors with their "degree" (Warnsdorff's heuristic)
            neighbors = []
            for (dr, dc) in self.ACTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited:
                        # Count how many unvisited neighbors this neighbor has
                        degree = self._count_unvisited_neighbors(nr, nc, visited | {(nr, nc)})
                        neighbors.append((degree, nr, nc))
            
            if not neighbors:
                # Dead end - backtrack with some randomness
                if len(path) <= 1:
                    return None
                
                # Backtrack
                last = path.pop()
                visited.remove(last)
                continue
            
            # Sort by degree (Warnsdorff: prefer cells with fewer exits)
            neighbors.sort(key=lambda x: x[0])
            
            # Add some randomness among equally good choices
            min_degree = neighbors[0][0]
            best_neighbors = [(d, r, c) for d, r, c in neighbors if d == min_degree]
            
            idx = self.rng.randint(0, len(best_neighbors))
            _, nr, nc = best_neighbors[idx]
            
            path.append((nr, nc))
            visited.add((nr, nc))
        
        return path if len(path) == total_cells else None
    
    def _generate_partial_path(self, length: int) -> list[tuple[int, int]] | None:
        """Generate a partial path (not visiting all cells)."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            start_r = self.rng.randint(0, self.size)
            start_c = self.rng.randint(0, self.size)
            
            path = [(start_r, start_c)]
            visited = {(start_r, start_c)}
            
            while len(path) < length:
                r, c = path[-1]
                
                neighbors = []
                for (dr, dc) in self.ACTIONS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if (nr, nc) not in visited:
                            neighbors.append((nr, nc))
                
                if not neighbors:
                    if len(path) == 1:
                        break
                    path.pop()
                    continue
                
                idx = self.rng.randint(0, len(neighbors))
                nr, nc = neighbors[idx]
                path.append((nr, nc))
                visited.add((nr, nc))
            
            if len(path) == length:
                return path
        
        return None
    
    def reset_solvable(self, seed: int = None) -> tuple[ZipResult, list[str]]:
        """
        Reset with a guaranteed solvable board.
        Generates Hamiltonian path (all cells) but places only num_count numbers.
        Returns (result, solution) where solution is list of moves to visit all cells.
        """
        if seed:
            self.rng = np.random.RandomState(seed)
        
        self._reset_state()
        
        total_cells = self.size * self.size
        
        # Generate FULL Hamiltonian path (visits all cells)
        path = self._generate_solvable_path(total_cells)
        if path is None:
            # Fallback to random (rare)
            return self.reset(seed), []
        
        # Place numbers at evenly distributed positions along the path
        # For example: if path=36 and num_count=10, place at indices 0, 4, 8, 12, ...
        step = (total_cells - 1) // (self.num_count - 1) if self.num_count > 1 else 1
        
        number_indices = []
        for i in range(self.num_count):
            idx = min(i * step, total_cells - 1)
            # Make sure last number is at the end
            if i == self.num_count - 1:
                idx = total_cells - 1
            number_indices.append(idx)
        
        # Place numbers at selected positions
        for num, idx in enumerate(number_indices, start=1):
            r, c = path[idx]
            self.board[r, c] = num
            self.number_pos[num] = (r, c)
        
        # Calculate solution moves (full path)
        solution = []
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            dr, dc = r2 - r1, c2 - c1
            
            for name, (adr, adc) in self.ACTIONS.items():
                if (adr, adc) == (dr, dc):
                    solution.append(name)
                    break
        
        # Set initial state
        self.current_pos = self.number_pos[1]
        self.visited = {self.current_pos}
        
        return ZipResult(
            observation=self._get_obs(),
            reward=0.0,
            done=False,
            info={"message": "Solvable game started", "has_solution": True}
        ), solution
    
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
