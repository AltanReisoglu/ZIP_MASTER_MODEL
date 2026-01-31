"""
ZIP Game Training - Standard Version (No Unsloth)

Uses standard Transformers + PEFT + TRL for GPU training.
Works with any CUDA-enabled PyTorch installation.
"""

from __future__ import annotations

import argparse
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from zip_env import ZipEnv, ZipAction, ZipObservation

SYSTEM_PROMPT = """You are an AI playing the ZIP game.

## Game Rules:
1. The board has numbers from 1 to N
2. Starting from number 1, you must reach all numbers in sequence
3. At each step you can only move: up, down, left, right
4. Each cell can only be visited once
5. Reach all numbers in order

## Response Format:
Do your thinking inside <think></think> tags.
Then give your answer: <answer>[up/down/left/right]</answer>
"""


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags or fallback patterns."""
    match = re.search(r'<answer>\s*\[?(up|down|left|right)\]?\s*</answer>', text.lower())
    if match:
        return match.group(1)
    match = re.search(r'\[(up|down|left|right)\]', text.lower())
    if match:
        return match.group(1)
    for action in ["up", "down", "left", "right"]:
        if action in text.lower():
            return action
    return ""


def build_prompt(obs: ZipObservation) -> str:
    """Build prompt for the model."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions) if obs.legal_actions else "none"
    return f"""## Board State:
{board_text}

## Available Moves: [{legal}]

Choose the next move:"""


def create_reward_function(env: ZipEnv):
    """Create a reward function for GRPO training."""
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            reward = 0.0
            action = extract_answer(completion)
            if "<answer>" in completion.lower():
                reward += 0.3
            if "<think>" in completion.lower():
                reward += 0.2
            if action in ["up", "down", "left", "right"]:
                reward += 1.0
            else:
                reward -= 2.0
            legal_match = re.search(r'Available Moves:\s*\[(.*?)\]', prompt)
            if legal_match:
                legal_str = legal_match.group(1)
                legal_actions = [a.strip() for a in legal_str.split(",") if a.strip()]
                if action in legal_actions:
                    reward += 1.0
                elif action and legal_actions:
                    reward -= 0.5
            rewards.append(reward)
        return rewards
    return reward_fn


def generate_game_prompts(env: ZipEnv, num_samples: int = 1000) -> list[dict]:
    """Generate training prompts from random game states."""
    samples = []
    for _ in range(num_samples):
        result = env.reset(seed=random.randint(0, 100000))
        obs = result.observation
        num_moves = random.randint(0, 15)
        for _ in range(num_moves):
            if result.done or not obs.legal_actions:
                break
            action = ZipAction(random.choice(obs.legal_actions))
            result = env.step(action)
            obs = result.observation
        if not result.done and obs.legal_actions:
            prompt = build_prompt(obs)
            samples.append({
                "prompt": prompt,
                "legal_actions": obs.legal_actions,
                "target": obs.current_target,
            })
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Game GRPO Training (Standard)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="./zip_output")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("ZIP Game Training - Standard Version (No Unsloth)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    print(f"\n[1/4] Loading model: {args.model_id}")
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
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
    
    print("\n[2/4] Creating environment and dataset")
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    print(f"Generating {args.dataset_size} training samples...")
    samples = generate_game_prompts(env, args.dataset_size)
    
    def format_sample(sample):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sample["prompt"]}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompts = [format_sample(s) for s in samples]
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset size: {len(dataset)}")
    
    print("\n[3/4] Setting up GRPO trainer")
    from trl import GRPOConfig, GRPOTrainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grpo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_fn = create_reward_function(env)
    
    training_args = GRPOConfig(
        output_dir=str(output_dir), learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs, num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens, temperature=0.7, top_p=0.9,
        logging_steps=10, report_to="none", save_strategy="steps", save_steps=args.save_steps,
        optim="adamw_8bit", warmup_ratio=0.1, weight_decay=0.01, max_grad_norm=1.0,
        seed=3407, bf16=False, fp16=True,
    )
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=training_args, train_dataset=dataset, reward_funcs=reward_fn)
    
    print("\n[4/4] Starting training")
    print("-" * 60)
    print(f"Model: {args.model_id}\nLoRA rank: {args.lora_rank}\n4-bit quantization: {args.use_4bit}")
    print(f"Dataset size: {len(dataset)}\nBatch size: {args.per_device_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}\nGenerations per prompt: {args.num_generations}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    model.save_pretrained(str(output_dir / "lora_adapters"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapters"))
    print(f"\n{'='*60}\nTraining completed!\nModel saved to: {output_dir / 'final'}\nLoRA adapters: {output_dir / 'lora_adapters'}\n{'='*60}")


if __name__ == "__main__":
    main()
