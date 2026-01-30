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

# Local imports
from zip_env import ZipEnv, ZipAction, ZipObservation,Zip


# ============== SYSTEM PROMPT ==============

SYSTEM_PROMPT = """Sen ZIP oyunu oynayan bir AI'sın.

## Oyun Kuralları:
1. Board üzerinde 1'den N'e kadar numaralar var
2. 1 numarasından başlayıp sırayla tüm numaralara ulaşmalısın
3. Her adımda sadece up, down, left, right hareket yapabilirsin
4. Bir hücre sadece bir kez ziyaret edilebilir
5. Tüm numaralara sırayla ulaş

## Cevap Formatı:
Düşünmeni <think></think> tagları içinde yap.
Sonra cevabını ver: <answer>[up/down/left/right]</answer>
"""


# ============== PARSING ==============

def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags or fallback patterns."""
    # Try <answer>action</answer> format
    match = re.search(r'<answer>\s*\[?(up|down|left|right)\]?\s*</answer>', text.lower())
    if match:
        return match.group(1)
    
    # Try [action] format
    match = re.search(r'\[(up|down|left|right)\]', text.lower())
    if match:
        return match.group(1)
    
    # Direct keyword search
    for action in ["up", "down", "left", "right"]:
        if action in text.lower():
            return action
    
    return ""


def build_prompt(obs: ZipObservation) -> str:
    """Build prompt for the model."""
    board_text = obs.to_text()
    legal = ", ".join(obs.legal_actions) if obs.legal_actions else "yok"
    
    return f"""## Board Durumu:
{board_text}

## Yapılabilir Hareketler: [{legal}]

Bir sonraki hareketi seç:"""


# ============== REWARD FUNCTION ==============

def create_reward_function(env: ZipEnv):
    """Create a reward function for GRPO training."""
    
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            reward = 0.0
            
            # Parse the prompt to understand current state
            action = extract_answer(completion)
            
            # Format rewards
            if "<answer>" in completion.lower():
                reward += 0.3
            if "<think>" in completion.lower():
                reward += 0.2
            
            # Action validity
            if action in ["up", "down", "left", "right"]:
                reward += 1.0
            else:
                reward -= 2.0  # Strong penalty for invalid
            
            # Simulate: Parse legal moves from prompt
            legal_match = re.search(r'Yapılabilir Hareketler:\s*\[(.*?)\]', prompt)
            if legal_match:
                legal_str = legal_match.group(1)
                legal_actions = [a.strip() for a in legal_str.split(",") if a.strip()]
                
                if action in legal_actions:
                    reward += 1.0  # Chose from legal actions
                elif action and legal_actions:
                    reward -= 0.5  # Chose illegal action
            
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


# ============== DATASET GENERATION ==============

def generate_game_prompts(env: ZipEnv, num_samples: int = 1000) -> list[dict]:
    """Generate training prompts from random game states."""
    samples = []
    
    for _ in range(num_samples):
        # Reset environment with random seed
        result = env.reset(seed=random.randint(0, 100000))
        obs = result.observation
        
        # Play some random moves to get varied states
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


# ============== MAIN TRAINING ==============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIP Game GRPO Training (Standard)")
    
    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    
    # Environment
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--num-count", type=int, default=10)
    
    # Training
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=32)
    
    # Quantization
    parser.add_argument("--use-4bit", action="store_true", default=True)
    
    # Checkpoints
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="./zip_output")
    
    # Debug
    parser.add_argument("--debug", action="store_true", default=False)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 60)
    print("ZIP Game Training - Standard Version (No Unsloth)")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    # ===== 1. Load Model =====
    print(f"\n[1/4] Loading model: {args.model_id}")
    
    # 4-bit quantization config - use float16 consistently
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
    
    # Prepare for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
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
    
    # ===== 2. Setup Environment =====
    print("\n[2/4] Creating environment and dataset")
    
    env = ZipEnv(size=args.board_size, num_count=args.num_count)
    
    # Generate prompts from game states
    print(f"Generating {args.dataset_size} training samples...")
    samples = generate_game_prompts(env, args.dataset_size)
    
    # Format as dataset
    def format_sample(sample):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    prompts = [format_sample(s) for s in samples]
    dataset = Dataset.from_dict({"prompt": prompts})
    
    print(f"Dataset size: {len(dataset)}")
    
    # ===== 3. Setup GRPO Trainer =====
    print("\n[3/4] Setting up GRPO trainer")
    
    from trl import GRPOConfig, GRPOTrainer
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grpo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create reward function
    reward_fn = create_reward_function(env)
    
    # GRPO Config
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        
        # Training params
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        
        # Generation params
        temperature=0.7,
        top_p=0.9,
        
        # Logging
        logging_steps=10,
        report_to="none",
        
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        
        # Optimization
        optim="adamw_8bit",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Misc
        seed=3407,
        bf16=False,
        fp16=True,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )
    
    # ===== 4. Train =====
    print("\n[4/4] Starting training")
    print("-" * 60)
    print(f"Model: {args.model_id}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"4-bit quantization: {args.use_4bit}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.per_device_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    
    # Also save LoRA adapters separately
    model.save_pretrained(str(output_dir / "lora_adapters"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapters"))
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Model saved to: {output_dir / 'final'}")
    print(f"LoRA adapters: {output_dir / 'lora_adapters'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
