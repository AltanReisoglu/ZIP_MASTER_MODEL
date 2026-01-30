"""
ZIP Game Training with TRL GRPO + Unsloth

Based on TRL OpenEnv examples (wordle.py, sudoku.py) and Unsloth GRPO patterns.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from trl.experimental.openenv import generate_rollout_completions
# Local imports
from zip_env import ZipEnv, ZipAction, ZipObservation, ZipResult


# ============== SYSTEM PROMPT ==============

SYSTEM_PROMPT = """Sen ZIP oyunu oynayan bir AI'sın.

## Oyun Kuralları:
1. 6x6 board üzerinde 1'den 10'a kadar numaralar var
2. 1 numarasından başlayıp sırayla tüm numaralara ulaşmalısın
3. Her adımda sadece up, down, left, right hareket yapabilirsin
4. Bir hücre sadece bir kez ziyaret edilebilir
5. Tüm numaralara sırayla ulaş

## Cevap Formatı:
Sadece şu formatı kullan: [action]
Örnek: [up], [down], [left], [right]

Düşün ve en iyi hareketi seç."""


# ============== PARSING ==============

def extract_action(text: str) -> str:
    """Extract action [up/down/left/right] from text."""
    # Try [action] format first
    match = re.search(r'\[(up|down|left|right)\]', text.lower())
    if match:
        return match.group(1)
    
    # Try ACTION: format
    if "action:" in text.lower():
        parts = text.lower().split("action:")
        if len(parts) > 1:
            action = parts[1].strip().split()[0].strip(".,!?[]")
            if action in ["up", "down", "left", "right"]:
                return action
    
    # Try direct keyword
    for action in ["up", "down", "left", "right"]:
        if action in text.lower():
            return action
    
    return ""


def make_prompt(obs: ZipObservation, move_history: list[str] = None) -> str:
    """Create a compact prompt for the model."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions) if obs.legal_actions else "None"
    
    history_text = ""
    if move_history and len(move_history) > 0:
        recent = move_history[-5:]  # Show last 5 moves only
        history_text = f"\n\nSon hareketler: {', '.join(recent)}"
    
    return f"""{SYSTEM_PROMPT}

## Board Durumu:
{board_text}

## Yapılabilir Hareketler: [{legal}]{history_text}

Bir sonraki hareketi seç:"""


# ============== REWARD FUNCTIONS ==============

def reward_valid_move(completions: list[str], **kwargs) -> list[float]:
    """Reward for making valid moves."""
    rewards = kwargs.get("valid_move_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_target_reached(completions: list[str], **kwargs) -> list[float]:
    """Reward for reaching the target number."""
    rewards = kwargs.get("target_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_progress(completions: list[str], **kwargs) -> list[float]:
    """Reward for overall progress in the game."""
    rewards = kwargs.get("progress_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_win(completions: list[str], **kwargs) -> list[float]:
    """Reward for winning the game."""
    rewards = kwargs.get("win_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_repetition(completions: list[str], **kwargs) -> list[float]:
    """Penalty for repeating moves."""
    rewards = kwargs.get("repetition_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ============== ROLLOUT FUNCTION ==============

def rollout_once(
    trainer,
    env: ZipEnv,
    tokenizer,
    system_prompt: str,
    max_turns: int = 50,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Play one complete game episode.
    Returns data for the LAST turn only (for efficient backprop).
    """
    
    
    result = env.reset()
    observation = result.observation
    
    # Store only last turn for backprop (efficient!)
    last_turn_data: dict | None = None
    
    # Track rewards
    valid_move_scores: list[float] = []
    target_scores: list[float] = []
    progress_scores: list[float] = []
    win_scores: list[float] = []
    repetition_scores: list[float] = []
    
    move_counts: defaultdict[str, int] = defaultdict(int)
    move_history: list[str] = []
    
    initial_target = observation.current_target
    max_target_reached = initial_target
    won = False
    
    for turn in range(max_turns):
        if result.done:
            break
        
        # Build prompt
        user_prompt = make_prompt(observation, move_history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        if debug:
            print(f"\n{'='*60}")
            print(f"STEP {turn + 1}")
            print(f"{'='*60}")
            print(f"PROMPT:\n{user_prompt[:500]}...")
        
        # Generate completion
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        
        # Store only this turn's data (replace previous)
        last_turn_data = {
            "prompt_ids": rollout_outputs["prompt_ids"],
            "completion_ids": rollout_outputs["completion_ids"],
            "logprobs": rollout_outputs["logprobs"],
        }
        
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        
        # Extract action
        action_str = extract_action(completion_text)
        
        if debug:
            print(f"MODEL OUTPUT: {completion_text}")
            print(f"EXTRACTED ACTION: {action_str}")
        
        # Calculate repetition penalty BEFORE move
        previous_occurrences = move_counts[action_str]
        move_counts[action_str] += 1
        
        if previous_occurrences > 0:
            # Exponential penalty: -2^(n-1) capped at -5
            repetition_score = -min(2 ** (previous_occurrences - 1), 5.0)
        else:
            repetition_score = 0.0
        
        # Step environment
        if action_str and action_str in observation.legal_actions:
            action = ZipAction(action_str)
            result = env.step(action)
            move_history.append(action_str)
            
            # Valid move reward
            valid_score = 1.0
            
            # Check if target was reached
            new_obs = result.observation
            if new_obs.current_target > observation.current_target:
                target_score = 2.0  # Bonus for reaching target
                max_target_reached = new_obs.current_target
            else:
                target_score = 0.0
            
            observation = new_obs
        else:
            # Invalid move - penalty
            valid_score = -1.0
            target_score = 0.0
            
            # Try random legal action as fallback
            if observation.legal_actions:
                import random
                fallback = random.choice(observation.legal_actions)
                action = ZipAction(fallback)
                result = env.step(action)
                move_history.append(f"({fallback})")
                observation = result.observation
        
        # Check for win
        if result.info.get("win"):
            win_score = 10.0
            won = True
        elif result.info.get("stuck"):
            win_score = -2.0
        else:
            win_score = 0.0
        
        valid_move_scores.append(valid_score)
        target_scores.append(target_score)
        win_scores.append(win_score)
        repetition_scores.append(repetition_score)
        
        if debug:
            print(f"VALID: {valid_score}, TARGET: {target_score}, WIN: {win_score}")
    
    # Calculate aggregate rewards
    valid_reward = sum(valid_move_scores) / max(len(valid_move_scores), 1)
    target_reward = sum(target_scores) / max(len(target_scores), 1)
    win_reward = win_scores[-1] if win_scores else 0.0
    repetition_reward = sum(repetition_scores) / max(len(repetition_scores), 1)
    
    # Progress reward: how many targets we reached
    total_targets = env.num_count
    progress_reward = (max_target_reached - 1) / (total_targets - 1) if total_targets > 1 else 0.0
    
    # Get final data
    if last_turn_data:
        prompt_ids = last_turn_data["prompt_ids"]
        completion_ids = last_turn_data["completion_ids"]
        logprobs = last_turn_data["logprobs"]
    else:
        prompt_ids = []
        completion_ids = []
        logprobs = []
    
    total_tokens = len(prompt_ids) + len(completion_ids)
    print(
        f"Episode: valid={valid_reward:.2f}, target={target_reward:.2f}, "
        f"progress={progress_reward:.2f}, win={win_reward:.2f}, "
        f"rep={repetition_reward:.2f}, tokens={total_tokens}, won={won}"
    )
    
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "valid_move_reward": valid_reward,
        "target_reward": target_reward,
        "progress_reward": progress_reward,
        "win_reward": win_reward,
        "repetition_reward": repetition_reward,
    }


# ============== MAIN TRAINING ==============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Game GRPO Training")
    
    # Model
    parser.add_argument("--model-id", default="unsloth/Ministral-3B-Instruct-2503")
    
    # Environment
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=50)
    
    # Training
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-rank", type=int, default=32)
    
    # Checkpoints
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    
    # Logging
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("ZIP Game Training with TRL GRPO + Unsloth")
    print("=" * 60)
    
    # Setup Unsloth model
    print(f"\n[1/4] Loading model: {args.model_id}")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM-like fast inference
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup environment
    print("\n[2/4] Creating environment")
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "prompt": ["Play the ZIP game."] * args.dataset_size
    })
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"./zip_output/grpo-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GRPO Config
    print("\n[3/4] Setting up GRPO trainer")
    
    from trl import GRPOConfig, GRPOTrainer
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_interval,
        # vLLM settings for Unsloth
        use_vllm=True,
        vllm_mode="colocate",
    )
    
    # Define rollout function
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_valid = []
        all_target = []
        all_progress = []
        all_win = []
        all_repetition = []
        
        for _ in prompts:
            episode = rollout_once(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                system_prompt=SYSTEM_PROMPT,
                max_turns=args.max_turns,
                debug=args.debug,
            )
            
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_valid.append(episode["valid_move_reward"])
            all_target.append(episode["target_reward"])
            all_progress.append(episode["progress_reward"])
            all_win.append(episode["win_reward"])
            all_repetition.append(episode["repetition_reward"])
        
        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "valid_move_reward": all_valid,
            "target_reward": all_target,
            "progress_reward": all_progress,
            "win_reward": all_win,
            "repetition_reward": all_repetition,
        }
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_valid_move,     # Learn to make valid moves
            reward_target_reached, # Learn to reach targets
            reward_progress,       # Learn to fill more of the board
            reward_win,            # Learn to complete the game
            reward_repetition,     # Avoid repeating moves
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )
    
    # Train
    print("\n[4/4] Starting training")
    print("-" * 60)
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max turns per episode: {args.max_turns}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    print(f"\nTraining completed! Model saved to: {final_path}")


if __name__ == "__main__":
    main()