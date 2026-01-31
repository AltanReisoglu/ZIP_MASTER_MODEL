"""
ZIP Game Inference - Test trained model

Enhanced version with solvable board testing and detailed statistics.
"""

from __future__ import annotations

import argparse
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from zip_env import ZipEnv, ZipAction, ZipObservation


# System prompt matching expert training
SYSTEM_PROMPT = """You are an expert ZIP game player. Connect numbers 1 to N in order by visiting ALL cells.

Rules:
- Move: up, down, left, right
- Visit each cell exactly once
- Reach all numbers in sequence

Think about the optimal path, then respond with: <answer>direction</answer>"""


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
    """Create prompt from observation (inference version - no hints)."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions)
    
    return f"""{SYSTEM_PROMPT}

## Current Board:
{board_text}

## Legal Moves: [{legal}]

Your move:"""


def load_trained_model(model_path: str, base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
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


def play_game(model, tokenizer, env: ZipEnv, max_turns: int = 40, verbose: bool = True, 
              use_solvable: bool = False, temperature: float = 0.1):
    """Play a complete game using the trained model."""
    
    solution = []
    if use_solvable:
        result, solution = env.reset_solvable()
    else:
        result = env.reset()
    
    obs = result.observation
    
    if verbose:
        print("=" * 60)
        print(f"ZIP GAME - AI Player {'(Solvable Board)' if use_solvable else ''}")
        print("=" * 60)
        print(f"\nInitial board:")
        print(obs.to_text())
        if solution:
            print(f"\n[Solution available: {len(solution)} moves]")
    
    move_history = []
    valid_moves = 0
    illegal_moves = 0
    optimal_moves = 0  # Moves that match the solution
    
    for turn in range(max_turns):
        if result.done or not obs.legal_actions:
            break
        
        # Expected optimal move (if we have solution)
        expected_optimal = solution[turn] if turn < len(solution) else None
        
        # Generate prompt
        prompt = make_prompt(obs)
        
        # Get model response
        response = generate_response(model, tokenizer, prompt, temperature=temperature)
        
        # Extract action
        action_str = extract_action(response)
        
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
            print(f"Model: {response[:80]}...")
            print(f"Action: {action_str} | Legal: {obs.legal_actions}", end="")
            if expected_optimal:
                print(f" | Optimal: {expected_optimal}")
            else:
                print()
        
        # Execute action
        if action_str and action_str in obs.legal_actions:
            action = ZipAction(action_str)
            result = env.step(action)
            move_history.append(action_str)
            obs = result.observation
            valid_moves += 1
            
            # Check if optimal
            if action_str == expected_optimal:
                optimal_moves += 1
                if verbose:
                    print(f"[✓] Optimal move! | Reward: {result.reward:.2f}")
            else:
                if verbose:
                    print(f"[OK] Valid move | Reward: {result.reward:.2f}")
        else:
            illegal_moves += 1
            if verbose:
                print(f"[X] Invalid action: '{action_str}'")
            
            # Random fallback
            if obs.legal_actions:
                fallback = random.choice(obs.legal_actions)
                action = ZipAction(fallback)
                result = env.step(action)
                move_history.append(f"({fallback})")
                obs = result.observation
    
    # Check win condition
    won = result.done and obs.current_target > env.num_count
    coverage = len(obs.visited) / (env.size ** 2)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"GAME OVER - {'🏆 WON!' if won else '❌ Lost'}")
        print(f"Valid: {valid_moves} | Illegal: {illegal_moves} | Coverage: {coverage:.1%}")
        if solution:
            print(f"Optimal moves: {optimal_moves}/{len(solution)} ({100*optimal_moves/max(1,len(solution)):.0f}%)")
        print(f"Moves: {' → '.join(move_history[:20])}{'...' if len(move_history) > 20 else ''}")
        print("=" * 60)
    
    return {
        "won": won,
        "valid_moves": valid_moves,
        "illegal_moves": illegal_moves,
        "optimal_moves": optimal_moves,
        "total_solution_moves": len(solution),
        "coverage": coverage,
        "move_history": move_history
    }


def run_benchmark(model, tokenizer, args):
    """Run comprehensive benchmark on both random and solvable boards."""
    
    results = {
        "random": {"wins": 0, "total": 0, "valid": 0, "illegal": 0},
        "solvable": {"wins": 0, "total": 0, "valid": 0, "illegal": 0, "optimal": 0, "solution_total": 0}
    }
    
    num_each = args.num_games // 2
    
    # Test on random boards
    print(f"\n{'='*60}")
    print(f"Testing on {num_each} RANDOM boards...")
    print("="*60)
    
    for i in range(num_each):
        env = ZipEnv(size=args.board_size, num_count=args.num_count)
        game_result = play_game(model, tokenizer, env, max_turns=args.max_turns, 
                               verbose=args.verbose, use_solvable=False)
        
        results["random"]["total"] += 1
        results["random"]["valid"] += game_result["valid_moves"]
        results["random"]["illegal"] += game_result["illegal_moves"]
        if game_result["won"]:
            results["random"]["wins"] += 1
        
        status = "Won" if game_result["won"] else "Lost"
        print(f"  Game {i+1}: {status} | Valid: {game_result['valid_moves']}, Illegal: {game_result['illegal_moves']}")
    
    # Test on solvable boards
    print(f"\n{'='*60}")
    print(f"Testing on {num_each} SOLVABLE boards...")
    print("="*60)
    
    for i in range(num_each):
        env = ZipEnv(size=args.board_size, num_count=args.num_count)
        game_result = play_game(model, tokenizer, env, max_turns=args.max_turns, 
                               verbose=args.verbose, use_solvable=True)
        
        results["solvable"]["total"] += 1
        results["solvable"]["valid"] += game_result["valid_moves"]
        results["solvable"]["illegal"] += game_result["illegal_moves"]
        results["solvable"]["optimal"] += game_result["optimal_moves"]
        results["solvable"]["solution_total"] += game_result["total_solution_moves"]
        if game_result["won"]:
            results["solvable"]["wins"] += 1
        
        status = "Won" if game_result["won"] else "Lost"
        opt_rate = game_result["optimal_moves"] / max(1, game_result["total_solution_moves"])
        print(f"  Game {i+1}: {status} | Valid: {game_result['valid_moves']}, Optimal: {opt_rate:.0%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ZIP Game Inference")
    parser.add_argument("--model-path", default="./zip_output/expert-2026-01-31_18-33-46/final")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--demo", action="store_true", default=True, help="Play one demo game first")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ZIP Game Inference - Model Evaluation")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_trained_model(args.model_path, args.base_model)
    
    # Demo game
    if args.demo:
        print("\n" + "=" * 70)
        print("Demo Game (Solvable Board)")
        print("=" * 70)
        env = ZipEnv(size=args.board_size, num_count=args.num_count)
        play_game(model, tokenizer, env, max_turns=args.max_turns, verbose=True, use_solvable=True)
    
    # Run benchmark
    results = run_benchmark(model, tokenizer, args)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    r_rand = results["random"]
    r_solv = results["solvable"]
    
    print(f"\n📊 Random Boards ({r_rand['total']} games):")
    print(f"   Win Rate: {r_rand['wins']}/{r_rand['total']} ({100*r_rand['wins']/max(1,r_rand['total']):.1f}%)")
    print(f"   Legal Move Rate: {r_rand['valid']}/{r_rand['valid']+r_rand['illegal']} ({100*r_rand['valid']/max(1,r_rand['valid']+r_rand['illegal']):.1f}%)")
    
    print(f"\n📊 Solvable Boards ({r_solv['total']} games):")
    print(f"   Win Rate: {r_solv['wins']}/{r_solv['total']} ({100*r_solv['wins']/max(1,r_solv['total']):.1f}%)")
    print(f"   Legal Move Rate: {r_solv['valid']}/{r_solv['valid']+r_solv['illegal']} ({100*r_solv['valid']/max(1,r_solv['valid']+r_solv['illegal']):.1f}%)")
    print(f"   Optimal Move Rate: {r_solv['optimal']}/{r_solv['solution_total']} ({100*r_solv['optimal']/max(1,r_solv['solution_total']):.1f}%)")
    
    total_wins = r_rand['wins'] + r_solv['wins']
    total_games = r_rand['total'] + r_solv['total']
    
    print(f"\n🎯 Overall Win Rate: {total_wins}/{total_games} ({100*total_wins/max(1,total_games):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()