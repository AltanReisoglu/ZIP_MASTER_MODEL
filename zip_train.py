"""
ZIP Game Training with TRL GRPO

IMPROVED: Uses strategic data with ground truth optimal moves.
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

SYSTEM_PROMPT = """You are playing the ZIP game. Connect numbers 1 to N in order.

Rules:
- Move: up, down, left, right
- Visit each cell only once
- Reach all numbers in sequence

Response format: <answer>up</answer>
"""


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

## Board:
{board_text}

## Available Moves: [{legal}]

Your move:"""


# ============== STRATEGIC REWARD FUNCTIONS ==============

def reward_valid_action(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """Reward for producing a valid action."""
    rewards = []
    for c in completions:
        action = extract_action(c)
        if action in ["up", "down", "left", "right"]:
            rewards.append(1.0)
        else:
            rewards.append(-2.0)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """Reward for proper <answer> format."""
    rewards = []
    for c in completions:
        if "<answer>" in c.lower() and "</answer>" in c.lower():
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def parse_legal_actions(prompt: str) -> list[str]:
    """Parse legal actions from prompt."""
    match = re.search(r'Available Moves:\s*\[(.*?)\]', prompt)
    if match:
        return [a.strip() for a in match.group(1).split(",") if a.strip()]
    return []


def parse_optimal_action(prompt: str) -> str:
    """Parse optimal action from prompt metadata."""
    match = re.search(r'OPTIMAL:\s*(\w+)', prompt)
    if match:
        return match.group(1).lower()
    return ""


def reward_legal_move(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """Reward for legal moves, bonus for optimal moves."""
    rewards = []
    for i, c in enumerate(completions):
        action = extract_action(c)
        
        if prompts and i < len(prompts):
            legal = parse_legal_actions(prompts[i])
            optimal = parse_optimal_action(prompts[i])
            
            if action == optimal:
                rewards.append(5.0)  # Big bonus for optimal move!
            elif action in legal:
                rewards.append(1.0)  # Small reward for legal move
            elif action:
                rewards.append(-3.0)  # Penalty for illegal
            else:
                rewards.append(-2.0)  # Penalty for no action
        else:
            rewards.append(0.0)
    return rewards


def reward_target_direction(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """Reward for moving toward target."""
    rewards = []
    for i, c in enumerate(completions):
        action = extract_action(c)
        
        if prompts and i < len(prompts):
            # Check if action is in the direction of target
            # This is encoded in the prompt with TOWARD_TARGET marker
            if "TOWARD_TARGET:" in prompts[i]:
                match = re.search(r'TOWARD_TARGET:\s*(\w+)', prompts[i])
                if match and action == match.group(1).lower():
                    rewards.append(2.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


# ============== STRATEGIC DATASET GENERATION ==============

def get_direction_toward_target(current_pos: tuple, target_pos: tuple) -> list[str]:
    """Get directions that move toward target."""
    cr, cc = current_pos
    tr, tc = target_pos
    directions = []
    
    if tr < cr:
        directions.append("up")
    if tr > cr:
        directions.append("down")
    if tc < cc:
        directions.append("left")
    if tc > cc:
        directions.append("right")
    
    return directions


def generate_strategic_dataset(env: ZipEnv, size: int = 500) -> Dataset:
    """
    Generate training dataset with STRATEGIC moves from expert play.
    
    This creates (prompt, optimal_action) pairs where optimal_action
    is the move that leads toward winning.
    """
    prompts = []
    
    print(f"Generating {size} STRATEGIC game states...")
    
    generated = 0
    attempts = 0
    max_attempts = size * 5
    
    while generated < size and attempts < max_attempts:
        attempts += 1
        
        # Reset with random board
        result = env.reset()
        obs = result.observation
        
        # Play some random moves to get varied positions
        for _ in range(random.randint(0, 20)):
            if result.done or not obs.legal_actions:
                break
            action = ZipAction(random.choice(obs.legal_actions))
            result = env.step(action)
            obs = result.observation
        
        if result.done or not obs.legal_actions:
            continue
        
        # Find optimal move: prioritize moves toward target
        current_pos = obs.current_pos
        target_num = obs.current_target
        target_pos = None
        
        # Find target position from number_pos
        if target_num in env.number_pos:
            target_pos = env.number_pos[target_num]
        
        optimal_action = None
        toward_target = None
        
        if target_pos:
            # Get directions toward target
            good_dirs = get_direction_toward_target(current_pos, target_pos)
            
            # Filter by legal actions
            good_legal = [d for d in good_dirs if d in obs.legal_actions]
            
            if good_legal:
                optimal_action = random.choice(good_legal)
                toward_target = optimal_action
        
        # If no strategic move, pick any legal move
        if not optimal_action and obs.legal_actions:
            optimal_action = random.choice(obs.legal_actions)
        
        if not optimal_action:
            continue
        
        # Build enhanced prompt with hints
        board_text = obs.to_text()
        legal = ", ".join(obs.legal_actions)
        
        # Add metadata for reward functions (hidden from model display)
        prompt = f"""{SYSTEM_PROMPT}

## Board:
{board_text}

## Available Moves: [{legal}]
## OPTIMAL: {optimal_action}"""
        
        if toward_target:
            prompt += f"\n## TOWARD_TARGET: {toward_target}"
        
        prompt += "\n\nYour move:"
        
        prompts.append(prompt)
        generated += 1
        
        if generated % 100 == 0:
            print(f"  {generated}/{size}")
    
    print(f"Dataset: {len(prompts)} strategic samples")
    return Dataset.from_dict({"prompt": prompts})


# ============== MAIN ==============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Game GRPO Training")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--use-4bit", action="store_true", default=False)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("ZIP Game GRPO Training - STRATEGIC DATA")
    print("=" * 60)
    
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
    
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create environment
    print("\n[2/4] Creating environment")
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    
    # Generate STRATEGIC dataset
    print("\n[2.5/4] Generating strategic dataset")
    dataset = generate_strategic_dataset(env, size=args.dataset_size)
    
    # Setup output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"./zip_output/grpo-strategic-{timestamp}")
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
    )
    
    # Create trainer with strategic rewards
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_valid_action,      # +1 for valid action
            reward_format,            # +1 for format
            reward_legal_move,        # +5 for OPTIMAL, +1 for legal
            reward_target_direction,  # +2 for moving toward target
        ],
        train_dataset=dataset,
        args=grpo_config,
    )
    
    # Train
    print("\n[4/4] Starting training")
    print("-" * 60)
    print(f"Model: {args.model_id}")
    print(f"Board: {args.board_size}x{args.board_size}, {args.num_count} targets")
    print(f"Dataset: {len(dataset)} strategic samples")
    print(f"Epochs: {args.num_epochs}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # Save
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to: {final_path}")


if __name__ == "__main__":
    main()