"""
ZIP Game Training with Expert Trajectories

HIGH ACCURACY VERSION: Uses guaranteed solvable boards with optimal move sequences.
"""

from __future__ import annotations

import argparse
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from zip_env import ZipEnv, ZipAction, ZipObservation

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


def make_prompt(obs: ZipObservation, optimal_action: str = None, solution_remaining: int = None) -> str:
    """Create prompt from observation with optional hints."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions)
    
    prompt = f"""{SYSTEM_PROMPT}

## Current Board:
{board_text}

## Legal Moves: [{legal}]"""
    
    # Add metadata for reward functions (model should not see these directly)
    if optimal_action:
        prompt += f"\n## OPTIMAL: {optimal_action}"
    
    if solution_remaining is not None:
        prompt += f"\n## REMAINING_STEPS: {solution_remaining}"
    
    prompt += "\n\nYour move:"
    
    return prompt


# ============== EXPERT REWARD FUNCTIONS ==============

def reward_optimal_move(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    Strong reward for choosing the optimal move from the solution.
    
    +10.0 -> Optimal move (from solution path)
    +2.0  -> Legal but not optimal
    -5.0  -> Illegal move
    -3.0  -> No valid action extracted
    """
    rewards = []
    for i, c in enumerate(completions):
        action = extract_action(c)
        
        if prompts and i < len(prompts):
            # Parse optimal action from prompt
            optimal_match = re.search(r'OPTIMAL:\s*(\w+)', prompts[i])
            optimal = optimal_match.group(1).lower() if optimal_match else None
            
            # Parse legal actions
            legal_match = re.search(r'Legal Moves:\s*\[(.*?)\]', prompts[i])
            legal = [a.strip() for a in legal_match.group(1).split(",")] if legal_match else []
            
            if action == optimal:
                rewards.append(10.0)  # Perfect - following solution
            elif action in legal:
                rewards.append(2.0)   # Good - at least legal
            elif action in ["up", "down", "left", "right"]:
                rewards.append(-5.0)  # Bad - illegal move
            else:
                rewards.append(-3.0)  # Bad - no valid action
        else:
            rewards.append(0.0)
    
    return rewards


def reward_format_strict(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for proper <answer> format.
    
    +2.0 -> Perfect format
    +0.5 -> Has direction but wrong format
    -2.0 -> No recognizable format
    """
    rewards = []
    for c in completions:
        if re.search(r'<answer>\s*(up|down|left|right)\s*</answer>', c.lower()):
            rewards.append(2.0)
        elif any(d in c.lower() for d in ["up", "down", "left", "right"]):
            rewards.append(0.5)
        else:
            rewards.append(-2.0)
    return rewards


def reward_consistency(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    Reward for consistent behavior - picking one clear action.
    
    +1.0 -> Single clear action
    -1.0 -> Multiple or ambiguous actions
    """
    rewards = []
    for c in completions:
        text = c.lower()
        actions_found = sum(1 for d in ["up", "down", "left", "right"] if d in text)
        
        if actions_found == 1:
            rewards.append(1.0)
        elif actions_found > 1:
            rewards.append(-1.0)
        else:
            rewards.append(-1.0)
    
    return rewards


def reward_brevity(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for concise responses. Expert players don't need long explanations.
    
    +1.0 -> Short and clear (under 100 chars)
    +0.5 -> Medium length (100-200 chars)
    +0.0 -> Long response
    """
    rewards = []
    for c in completions:
        length = len(c)
        if length < 100:
            rewards.append(1.0)
        elif length < 200:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ============== EXPERT DATASET GENERATION ==============

def generate_expert_dataset(env: ZipEnv, num_games: int = 200) -> Dataset:
    """
    Generate training dataset from EXPERT play on solvable boards.
    
    Each sample contains:
    - prompt: Current game state
    - optimal: The correct move from the solution
    - legal: All legal moves at this state
    
    This provides ground truth for every step!
    """
    samples = []
    successful_games = 0
    total_attempts = 0
    max_attempts = num_games * 3
    
    print(f"Generating expert dataset from {num_games} solvable games...")
    
    while successful_games < num_games and total_attempts < max_attempts:
        total_attempts += 1
        
        # Get solvable board with solution
        try:
            result, solution = env.reset_solvable()
        except Exception as e:
            continue
        
        if not solution:
            continue
        
        obs = result.observation
        
        # Follow the solution and collect (state, action) pairs
        for step_idx, optimal_move in enumerate(solution):
            if result.done or not obs.legal_actions:
                break
            
            # Skip if optimal move is not in legal actions (shouldn't happen but safety check)
            if optimal_move not in obs.legal_actions:
                break
            
            # Calculate remaining steps in solution
            remaining = len(solution) - step_idx
            
            # Create training sample
            prompt = make_prompt(obs, optimal_action=optimal_move, solution_remaining=remaining)
            
            samples.append({
                "prompt": prompt,
                "optimal": optimal_move,
                "legal": obs.legal_actions.copy(),
                "step": step_idx,
                "total_steps": len(solution)
            })
            
            # Execute the move
            result = env.step(ZipAction(optimal_move))
            obs = result.observation
        
        # Count as successful if we got at least half the moves
        if len(solution) > 0:
            successful_games += 1
            
            if successful_games % 50 == 0:
                print(f"  {successful_games}/{num_games} games, {len(samples)} samples")
    
    print(f"\nDataset generated:")
    print(f"  - Games: {successful_games}")
    print(f"  - Total samples: {len(samples)}")
    print(f"  - Avg samples per game: {len(samples) / max(1, successful_games):.1f}")
    
    # Create HuggingFace dataset
    return Dataset.from_dict({
        "prompt": [s["prompt"] for s in samples]
    })


def generate_mixed_dataset(env: ZipEnv, num_expert: int = 150, num_random: int = 50) -> Dataset:
    """
    Generate mixed dataset: mostly expert, some random positions.
    
    This helps with generalization - model sees both optimal and suboptimal situations.
    """
    samples = []
    
    # Expert samples (following solution)
    print(f"Generating {num_expert} expert game samples...")
    expert_samples = 0
    attempts = 0
    
    while expert_samples < num_expert and attempts < num_expert * 3:
        attempts += 1
        
        try:
            result, solution = env.reset_solvable()
        except:
            continue
        
        if not solution:
            continue
        
        obs = result.observation
        
        for step_idx, optimal_move in enumerate(solution):
            if result.done or not obs.legal_actions:
                break
            if optimal_move not in obs.legal_actions:
                break
            
            remaining = len(solution) - step_idx
            prompt = make_prompt(obs, optimal_action=optimal_move, solution_remaining=remaining)
            samples.append({"prompt": prompt})
            
            result = env.step(ZipAction(optimal_move))
            obs = result.observation
        
        expert_samples += 1
        if expert_samples % 30 == 0:
            print(f"  Expert: {expert_samples}/{num_expert}")
    
    # Random position samples (to teach legal move selection)
    print(f"\nGenerating {num_random} random position samples...")
    random_samples = 0
    attempts = 0
    
    while random_samples < num_random and attempts < num_random * 5:
        attempts += 1
        
        result = env.reset()
        obs = result.observation
        
        # Take random number of moves
        for _ in range(random.randint(0, 25)):
            if result.done or not obs.legal_actions:
                break
            action = ZipAction(random.choice(obs.legal_actions))
            result = env.step(action)
            obs = result.observation
        
        if result.done or not obs.legal_actions:
            continue
        
        # Find best move toward target
        target_num = obs.current_target
        target_pos = env.number_pos.get(target_num)
        
        best_action = None
        if target_pos:
            cr, cc = obs.current_pos
            tr, tc = target_pos
            
            for action in obs.legal_actions:
                dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action]
                nr, nc = cr + dr, cc + dc
                
                # Check if this move gets closer to target
                old_dist = abs(tr - cr) + abs(tc - cc)
                new_dist = abs(tr - nr) + abs(tc - nc)
                
                if new_dist < old_dist:
                    best_action = action
                    break
        
        if not best_action:
            best_action = random.choice(obs.legal_actions)
        
        prompt = make_prompt(obs, optimal_action=best_action)
        samples.append({"prompt": prompt})
        random_samples += 1
    
    print(f"\nTotal samples: {len(samples)}")
    
    # Shuffle
    random.shuffle(samples)
    
    return Dataset.from_dict({
        "prompt": [s["prompt"] for s in samples]
    })


# ============== MAIN ==============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Expert Training")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=300, help="Number of expert games for dataset")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--use-4bit", action="store_true", default=False)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mixed-dataset", action="store_true", default=False,
                        help="Use mixed expert + random dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 70)
    print("ZIP Expert Training - HIGH ACCURACY MODE")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/4] Loading model: {args.model_id}")
    
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Larger LoRA for better learning
    lora_config = LoraConfig(
        r=args.lora_rank, 
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,  # Small dropout for regularization
        bias="none", 
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create environment
    print("\n[2/4] Creating environment")
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    
    # Generate expert dataset
    print("\n[2.5/4] Generating expert trajectory dataset")
    if args.mixed_dataset:
        dataset = generate_mixed_dataset(env, num_expert=args.num_games, num_random=args.num_games // 4)
    else:
        dataset = generate_expert_dataset(env, num_games=args.num_games)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"./zip_output/expert-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GRPO Config
    print("\n[3/4] Setting up GRPO trainer")
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_interval,
        report_to="none",
        # CPU compatibility
        bf16=False,
        fp16=False,
        # Better optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
    
    # Create trainer with expert reward functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_optimal_move,     # +10 for optimal, +2 for legal, -5 for illegal
            reward_format_strict,    # +2 for perfect format
            reward_consistency,      # +1 for single clear action
            reward_brevity,          # +1 for concise response
        ],
        train_dataset=dataset,
        args=grpo_config,
    )
    
    # Train
    print("\n[4/4] Starting training")
    print("-" * 70)
    print(f"Model: {args.model_id}")
    print(f"Board: {args.board_size}x{args.board_size}, {args.num_count} targets")
    print(f"Dataset: {len(dataset)} expert samples")
    print(f"Epochs: {args.num_epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output: {output_dir}")
    print("-" * 70)
    print("\nReward functions:")
    print("  - reward_optimal_move: +10 optimal, +2 legal, -5 illegal, -3 no action")
    print("  - reward_format_strict: +2 perfect format, +0.5 has direction, -2 no format")
    print("  - reward_consistency: +1 single action, -1 ambiguous")
    print("  - reward_brevity: +1 short, +0.5 medium, 0 long")
    print("\nTotal reward range: -11 to +14")
    print("-" * 70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted - saving current state...")
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n✓ Model saved to: {final_path}")
    
    print("\n" + "=" * 70)
    print("Training complete! Run inference to test:")
    print(f"  python inference.py --model-path {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
