"""
ZIP Game Inference - Test trained model
"""

from __future__ import annotations

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from zip_env import ZipEnv, ZipAction
from zip_train import make_prompt, extract_action, SYSTEM_PROMPT


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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
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
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def play_game(model, tokenizer, env: ZipEnv, max_turns: int = 30, verbose: bool = True):
    """Play a complete game using the trained model."""
    # Use fast random board
    result = env.reset()
    obs = result.observation
    
    if verbose:
        print("=" * 60)
        print("ZIP GAME - AI Player")
        print("=" * 60)
        print(f"\nInitial board:")
        print(obs.to_text())
    
    move_history = []
    valid_moves = 0
    illegal_moves = 0
    
    for turn in range(max_turns):
        if result.done or not obs.legal_actions:
            break
        
        # Generate prompt (matches zip_train.py format)
        prompt = make_prompt(obs)
        
        # Get model response
        response = generate_response(model, tokenizer, prompt)
        
        # Extract action
        action_str = extract_action(response)
        
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
            print(f"Model: {response[:100]}...")
            print(f"Action: {action_str} | Legal: {obs.legal_actions}")
        
        # Execute action
        if action_str and action_str in obs.legal_actions:
            action = ZipAction(action_str)
            result = env.step(action)
            move_history.append(action_str)
            obs = result.observation
            valid_moves += 1
            
            if verbose:
                print(f"[OK] Valid move | Reward: {result.reward:.2f}")
        else:
            illegal_moves += 1
            if verbose:
                print(f"[X] Invalid action!")
            
            # Random fallback
            if obs.legal_actions:
                fallback = random.choice(obs.legal_actions)
                action = ZipAction(fallback)
                result = env.step(action)
                move_history.append(f"({fallback})")
                obs = result.observation
    
    # Check win condition
    won = result.done and obs.current_target > env.num_count
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"GAME OVER - {'WON!' if won else 'Lost'}")
        print(f"Valid moves: {valid_moves}, Illegal: {illegal_moves}")
        print(f"Moves: {move_history}")
        print("=" * 60)
    
    return won, valid_moves, illegal_moves


def main():
    parser = argparse.ArgumentParser(description="ZIP Game Inference")
    parser.add_argument("--model-path", default="./zip_output/grpo-strategic-2026-01-31_13-47-18/final")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(args.model_path, args.base_model)
    
    # Play one verbose game
    print("\n" + "=" * 60)
    print("Playing demo game...")
    print("=" * 60)
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    play_game(model, tokenizer, env, max_turns=args.max_turns, verbose=True)
    
    # Run statistics
    print("\n" + "=" * 60)
    print(f"Running {args.num_games} games for statistics...")
    print("=" * 60)
    
    wins = 0
    total_valid = 0
    total_illegal = 0
    
    for i in range(args.num_games):
        env = ZipEnv(size=args.board_size, num_count=args.num_count)
        won, valid, illegal = play_game(model, tokenizer, env, max_turns=args.max_turns, verbose=args.verbose)
        
        if won:
            wins += 1
        total_valid += valid
        total_illegal += illegal
        print(f"Game {i+1}: {'Won' if won else 'Lost'} | Valid: {valid}, Illegal: {illegal}")
    
    print("\n" + "-" * 60)
    print(f"Win rate: {wins}/{args.num_games} ({wins/args.num_games*100:.1f}%)")
    print(f"Avg valid moves: {total_valid/args.num_games:.1f}")
    print(f"Avg illegal moves: {total_illegal/args.num_games:.1f}")
    print("-" * 60)


if __name__ == "__main__":
    main()