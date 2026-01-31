"""
ZIP Game Web UI - Interactive Game Interface

A FastAPI-based web interface to visualize and play the ZIP game with AI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import random
import re
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from zip_env import ZipEnv, ZipAction, ZipObservation

app = FastAPI(title="ZIP Game AI", description="Interactive ZIP Game Dashboard")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
game_state = {
    "env": None,
    "model": None,
    "tokenizer": None,
    "current_obs": None,
    "game_history": [],
    "stats": {
        "total_games": 0,
        "wins": 0,
        "valid_moves": 0,
        "illegal_moves": 0,
        "optimal_moves": 0
    },
    "is_playing": False,
    "solution": []
}

# System prompt matching expert training
SYSTEM_PROMPT = """You are an expert ZIP game player. Connect numbers 1 to N in order by visiting ALL cells.

Rules:
- Move: up, down, left, right
- Visit each cell exactly once
- Reach all numbers in sequence

Think about the optimal path, then respond with: <answer>direction</answer>"""


# Request/Response Models
class LoadModelRequest(BaseModel):
    model_path: str = "./zip_output/expert-2026-01-31_18-33-46/final"
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

class NewGameRequest(BaseModel):
    size: int = 6
    num_count: int = 8
    solvable: bool = True

class PlayerMoveRequest(BaseModel):
    direction: str

class AutoPlayRequest(BaseModel):
    delay: float = 0.5
    max_turns: int = 40


def extract_action(text: str) -> str:
    """Extract action from model output."""
    match = re.search(r'<answer>\s*(up|down|left|right)\s*</answer>', text.lower())
    if match:
        return match.group(1)
    for action in ["up", "down", "left", "right"]:
        if action in text.lower():
            return action
    return ""


def make_prompt(obs: ZipObservation) -> str:
    """Create prompt from observation."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions)
    
    return f"""{SYSTEM_PROMPT}

## Current Board:
{board_text}

## Legal Moves: [{legal}]

Your move:"""


def load_model(model_path: str, base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the trained LoRA model."""
    print(f"Loading base model: {base_model_id}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 48, temperature: float = 0.1) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def obs_to_dict(obs: ZipObservation) -> dict:
    """Convert observation to dictionary."""
    return {
        "board": obs.board,
        "current_pos": obs.current_pos,
        "current_target": obs.current_target,
        "visited": obs.visited,
        "legal_actions": obs.legal_actions,
        "text": obs.to_text()
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status():
    """Get current game status."""
    model_loaded = game_state["model"] is not None
    
    obs_data = None
    if game_state["current_obs"]:
        obs_data = obs_to_dict(game_state["current_obs"])
    
    return {
        "model_loaded": model_loaded,
        "is_playing": game_state["is_playing"],
        "stats": game_state["stats"],
        "observation": obs_data,
        "game_history": game_state["game_history"][-20:],
        "has_solution": len(game_state["solution"]) > 0
    }


@app.post("/api/load_model")
async def api_load_model(request: LoadModelRequest):
    """Load the AI model."""
    try:
        game_state["model"], game_state["tokenizer"] = load_model(
            request.model_path, request.base_model
        )
        return {"success": True, "message": "Model loaded successfully!"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/new_game")
async def new_game(request: NewGameRequest):
    """Start a new game."""
    env = ZipEnv(size=request.size, num_count=request.num_count)
    game_state["env"] = env
    game_state["game_history"] = []
    
    if request.solvable:
        result, solution = env.reset_solvable()
        game_state["solution"] = solution
    else:
        result = env.reset()
        game_state["solution"] = []
    
    game_state["current_obs"] = result.observation
    game_state["is_playing"] = True
    
    return {
        "success": True,
        "observation": obs_to_dict(result.observation),
        "solution_length": len(game_state["solution"])
    }


@app.post("/api/ai_move")
async def ai_move():
    """Let AI make a move."""
    if game_state["model"] is None:
        return {"success": False, "error": "Model not loaded"}
    
    if game_state["env"] is None:
        return {"success": False, "error": "No game started"}
    
    obs = game_state["current_obs"]
    if not obs.legal_actions:
        return {"success": False, "error": "No legal moves", "game_over": True}
    
    # Get move index for solution check
    move_index = len(game_state["game_history"])
    expected_optimal = game_state["solution"][move_index] if move_index < len(game_state["solution"]) else None
    
    # Generate AI move
    prompt = make_prompt(obs)
    response = generate_response(game_state["model"], game_state["tokenizer"], prompt)
    action_str = extract_action(response)
    
    move_data = {
        "turn": move_index + 1,
        "response": response[:100],
        "action": action_str,
        "expected": expected_optimal,
        "legal": obs.legal_actions.copy()
    }
    
    # Execute action
    if action_str and action_str in obs.legal_actions:
        action = ZipAction(action_str)
        result = game_state["env"].step(action)
        game_state["current_obs"] = result.observation
        game_state["stats"]["valid_moves"] += 1
        
        move_data["valid"] = True
        move_data["reward"] = result.reward
        move_data["optimal"] = (action_str == expected_optimal)
        
        if action_str == expected_optimal:
            game_state["stats"]["optimal_moves"] += 1
        
        # Check game end
        if result.done:
            won = result.observation.current_target > game_state["env"].num_count
            move_data["game_over"] = True
            move_data["won"] = won
            game_state["stats"]["total_games"] += 1
            if won:
                game_state["stats"]["wins"] += 1
    else:
        game_state["stats"]["illegal_moves"] += 1
        move_data["valid"] = False
        
        # Random fallback
        if obs.legal_actions:
            fallback = random.choice(obs.legal_actions)
            action = ZipAction(fallback)
            result = game_state["env"].step(action)
            game_state["current_obs"] = result.observation
            move_data["fallback"] = fallback
    
    game_state["game_history"].append(move_data)
    
    return {
        "success": True,
        "move": move_data,
        "observation": obs_to_dict(game_state["current_obs"]),
        "stats": game_state["stats"]
    }


@app.post("/api/player_move")
async def player_move(request: PlayerMoveRequest):
    """Human player makes a move."""
    if game_state["env"] is None:
        return {"success": False, "error": "No game started"}
    
    direction = request.direction
    obs = game_state["current_obs"]
    
    if direction not in obs.legal_actions:
        return {"success": False, "error": f"Illegal move: {direction}", "legal": obs.legal_actions}
    
    action = ZipAction(direction)
    result = game_state["env"].step(action)
    game_state["current_obs"] = result.observation
    
    move_data = {
        "turn": len(game_state["game_history"]) + 1,
        "action": direction,
        "valid": True,
        "reward": result.reward,
        "player": "human"
    }
    
    if result.done:
        won = result.observation.current_target > game_state["env"].num_count
        move_data["game_over"] = True
        move_data["won"] = won
    
    game_state["game_history"].append(move_data)
    
    return {
        "success": True,
        "move": move_data,
        "observation": obs_to_dict(result.observation)
    }


@app.post("/api/reset_stats")
async def reset_stats():
    """Reset game statistics."""
    game_state["stats"] = {
        "total_games": 0,
        "wins": 0,
        "valid_moves": 0,
        "illegal_moves": 0,
        "optimal_moves": 0
    }
    return {"success": True}


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ZIP Game Web UI")
    print("="*60)
    print("\n[*] Starting server at http://localhost:8000")
    print("[*] API docs at http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
