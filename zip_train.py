"""
ZIP Game Training with TRL GRPO

Based on TRL OpenEnv examples (wordle.py, sudoku.py) - no vLLM dependency.
Uses standard model.generate() for rollouts.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer


from zip_env import ZipEnv, ZipAction, ZipObservation, ZipResult

SYSTEM_PROMPT = """Sen ZIP oyunu oynayan bir AI'sın.

## Oyun Kuralları:
1. Board üzerinde 1'den N'e kadar numaralar var
2. 1 numarasından başlayıp sırayla tüm numaralara ulaşmalısın
3. Her adımda sadece up, down, left, right hareket yapabilirsin
4. Bir hücre sadece bir kez ziyaret edilebilir
5. Tüm numaralara sırayla ulaş

## ZORUNLU Cevap Formatı:
ÖNCE düşün, SONRA cevap ver!

<think>
- Şu anki pozisyonum nerede?
- Hedef numara nerede?
- Hangi yönler legal?
- En iyi hareket hangisi?
</think>
<answer>up</answer>

Sadece up, down, left, right kullan!
"""


# ============== PARSING ==============

def extract_action(text: str) -> str:
    """Extract action [up/down/left/right] from text."""
    
    match = re.search(r'<answer>(up|down|left|right)</answer>', text.lower())
    if match:
        return match.group(1)
    
    
    match = re.search(r'\[(up|down|left|right)\]', text.lower())
    if match:
        return match.group(1)
    
    
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
        recent = move_history[-5:]  
        history_text = f"\n\nSon hareketler: {', '.join(recent)}"
    
    return f"""{SYSTEM_PROMPT}

## Board Durumu:
{board_text}

## Yapılabilir Hareketler: [{legal}]{history_text}

Bir sonraki hareketi seç:"""



def reward_valid_action(completions: list[str], **kwargs) -> list[float]:
    """Reward for extracting a valid action from completion."""
    rewards = []
    for c in completions:
        action = extract_action(c)
        if action in ["up", "down", "left", "right"]:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """Reward for proper <answer> format."""
    rewards = []
    for c in completions:
        if "<answer>" in c.lower() and "</answer>" in c.lower():
            rewards.append(1.0)
        elif "<answer>" in c.lower():
            rewards.append(0.5)
        else:
            rewards.append(-1.0)
    return rewards


def reward_thinking(completions: list[str], **kwargs) -> list[float]:
    """Strong reward for quality thinking in <think> tags."""
    import re
    rewards = []
    for c in completions:
        c_lower = c.lower()
        
        # Check for think tags
        if "<think>" in c_lower and "</think>" in c_lower:
            match = re.search(r'<think>(.*?)</think>', c_lower, re.DOTALL)
            if match:
                content = match.group(1).strip()
                
                # Score based on quality indicators
                score = 0.5  # Base for having think tags
                
                # Bonus for mentioning directions
                if any(d in content for d in ['up', 'down', 'left', 'right']):
                    score += 0.5
                
                # Bonus for mentioning position or target
                if any(w in content for w in ['pozisyon', 'hedef', 'target', 'numara']):
                    score += 0.5
                
                # Bonus for reasoning words
                if any(w in content for w in ['çünkü', 'because', 'ise', 'olmalı', 'gitmeli']):
                    score += 0.5
                
                # Length bonus (but not too long)
                if 20 < len(content) < 200:
                    score += 0.5
                
                rewards.append(min(score, 2.0))  # Cap at 2.0
            else:
                rewards.append(0.0)
        else:
            rewards.append(-1.0)  # Penalty for no thinking
    return rewards


def reward_conciseness(completions: list[str], **kwargs) -> list[float]:
    """Reward for concise responses (not hitting max length)."""
    rewards = []
    for c in completions:
        length = len(c)
        if length < 200:
            rewards.append(1.0)  # Very concise
        elif length < 400:
            rewards.append(0.5)  # Reasonable
        else:
            rewards.append(-0.5)  # Too long (likely hit max tokens)
    return rewards


def reward_coverage(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    Reward for visiting more cells on the board.
    Higher reward for higher coverage.
    """
    rewards = []
    for c in completions:
        # Count visited cells mentioned in completion
        # For now, give reward based on valid action (more moves = more coverage)
        action = extract_action(c)
        if action in ["up", "down", "left", "right"]:
            rewards.append(0.5)  # Each valid move contributes to coverage
        else:
            rewards.append(-0.5)
    return rewards




def generate_solvable_dataset(env: ZipEnv, size: int = 100, save_path: str = None) -> Dataset:
    """
    Generate training dataset from solvable ZIP boards.
    Each entry contains a prompt (board state) and optional solution.
    """
    prompts = []
    solutions = []
    
    print(f"Generating {size} random boards...")
    
    for i in range(size):
        result = env.reset()  # Random board (fast!)
        obs = result.observation
        prompt = make_prompt(obs)
        
        prompts.append(prompt)
        solutions.append("")  # No pre-computed solution
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{size} boards")
    
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "solution": solutions,
    })
    
    
    if save_path:
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to: {save_path}")
    
    print(f"Dataset created: {len(dataset)} solvable boards")
    return dataset




def generate_completion(model, tokenizer, prompt_text: str, max_new_tokens: int = 128, temperature: float = 0.7) -> dict:
    """
    Generate completion using standard HuggingFace model.generate().
    Returns dict with prompt_ids, completion_ids, logprobs, text.
    """
    device = next(model.parameters()).device
    
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_ids = input_ids[0].tolist()
    
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    
    generated_ids = outputs.sequences[0]
    completion_ids = generated_ids[len(prompt_ids):].tolist()
    
    
    logprobs = []
    if outputs.scores:
        for i, score in enumerate(outputs.scores):
            if i < len(completion_ids):
                token_id = completion_ids[i]
                log_softmax = torch.log_softmax(score[0], dim=-1)
                logprobs.append(log_softmax[token_id].item())
    

    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": text,
    }




def rollout_once(
    model,
    tokenizer,
    env: ZipEnv,
    system_prompt: str,
    max_turns: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Play one complete game episode.
    Returns data for the LAST turn only (for efficient backprop).
    """
    
    result = env.reset()
    observation = result.observation
    
    
    last_turn_data: dict | None = None
    
    
    valid_move_scores: list[float] = []
    target_scores: list[float] = []
    progress_scores: list[float] = []
    win_scores: list[float] = []
    repetition_scores: list[float] = []
    format_scores: list[float] = []
    think_scores: list[float] = []
    
    move_counts: defaultdict[str, int] = defaultdict(int)
    move_history: list[str] = []
    
    initial_target = observation.current_target
    max_target_reached = initial_target
    won = False
    
    for turn in range(max_turns):
        if result.done:
            break
        
        
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
        
        
        rollout_outputs = generate_completion(
            model, tokenizer, prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        
        last_turn_data = {
            "prompt_ids": rollout_outputs["prompt_ids"],
            "completion_ids": rollout_outputs["completion_ids"],
            "logprobs": rollout_outputs["logprobs"],
        }
        
        completion_text = rollout_outputs["text"]
        
        
        action_str = extract_action(completion_text)
        
        if debug:
            print(f"MODEL OUTPUT: {completion_text}")
            print(f"EXTRACTED ACTION: {action_str}")
        
        # Calculate repetition penalty BEFORE move
        previous_occurrences = move_counts[action_str]
        move_counts[action_str] += 1

        if "<answer>" in completion_text:
            format_scores.append(1.0)
        else:
            format_scores.append(-1.0)

        if "<think>" in completion_text.lower() and "</think>" in completion_text.lower():
            think_scores.append(1.0)
        else:
            think_scores.append(-1.0)
        
        if previous_occurrences > 0:
            
            repetition_score = -min(2 ** (previous_occurrences - 1), 5.0)
        else:
            repetition_score = 0.0
        
        
        if action_str and action_str in observation.legal_actions:
            action = ZipAction(action_str)
            result = env.step(action)
            move_history.append(action_str)
            
            
            valid_score = 1.0
            
            
            new_obs = result.observation
            if new_obs.current_target > observation.current_target:
                target_score = 2.0  
                max_target_reached = new_obs.current_target
            else:
                target_score = 0.0
            
            observation = new_obs
        else:
            
            valid_score = -1.0
            target_score = 0.0
            
            
            if observation.legal_actions:
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
    format_reward = sum(format_scores) / max(len(format_scores), 1)
    think_reward = sum(think_scores) / max(len(think_scores), 1)
    
    
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
        "format_reward": format_reward,
        "think_reward": think_reward,
    }


# ============== MAIN TRAINING ==============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Game GRPO Training")
    
    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    
    # Environment
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=50)
    
    # Training
    parser.add_argument("--dataset-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-rank", type=int, default=32)
    
    # Quantization
    parser.add_argument("--use-4bit", action="store_true", default=False)
    
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
    print("ZIP Game Training with TRL GRPO")
    print("=" * 60)
    
    # Setup model
    print(f"\n[1/4] Loading model: {args.model_id}")
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training with LoRA
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Setup environment
    print("\n[2/4] Creating environment")
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    
    # Create solvable dataset
    print("\n[2.5/4] Generating solvable dataset")
    dataset = generate_solvable_dataset(env, size=args.dataset_size)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"./zip_output/grpo-{timestamp}")
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
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_interval,
        report_to="none",  # Disable WandB
    )
    
    # Define rollout function for GRPO
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_valid = []
        all_target = []
        all_progress = []
        all_win = []
        all_repetition = []
        all_format = []
        all_think = []
        
        for _ in prompts:
            episode = rollout_once(
                model=model,
                tokenizer=tokenizer,
                env=env,
                system_prompt=SYSTEM_PROMPT,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
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
            all_format.append(episode["format_reward"])
            all_think.append(episode["think_reward"])
        
        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "valid_move_reward": all_valid,
            "target_reward": all_target,
            "progress_reward": all_progress,
            "win_reward": all_win,
            "repetition_reward": all_repetition,
            "format_reward": all_format,
        }
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_valid_action,  # Extract valid action
            reward_format,        # Proper <answer> format
            reward_thinking,      # Use <think> tags
            reward_conciseness,   # Don't hit max tokens
            reward_coverage,      # Visit all cells
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )
    
    # Train
    print("\n[4/4] Starting training")
    print("-" * 60)
    print(f"Model: {args.model_id}")
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
    tokenizer.save_pretrained(str(final_path))
    print(f"\nTraining completed! Model saved to: {final_path}")


if __name__ == "__main__":
    main()