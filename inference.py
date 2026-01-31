"""
ZIP Game Inference - Test trained model
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from zip_env import ZipEnv, ZipAction
from zip_train import make_prompt, extract_action, SYSTEM_PROMPT


def load_trained_model(model_path: str, base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the trained LoRA model."""
    print(f"Loading base model: {base_model_id}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
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
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def play_game(model, tokenizer, env: ZipEnv, max_turns: int = 20, verbose: bool = True):
    """Play a complete game using the trained model."""
    result, solution = env.reset_solvable()
    obs = result.observation
    
    if verbose:
        print("=" * 60)
        print("ZIP GAME - AI Player")
        print("=" * 60)
        print(f"\nKnown solution: {solution}")
        print(f"\nInitial board:")
        print(obs.to_text())
    
    move_history = []
    
    for turn in range(max_turns):
        if result.done:
            break
        
        # Generate prompt
        prompt = make_prompt(obs, move_history)
        
        # Get model response
        response = generate_response(model, tokenizer, prompt)
        
        # Extract action
        action_str = extract_action(response)
        
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
            print(f"Model response:\n{response[:500]}")
            print(f"\nExtracted action: {action_str}")
            print(f"Legal actions: {obs.legal_actions}")
        
        # Execute action
        if action_str and action_str in obs.legal_actions:
            action = ZipAction(action_str)
            result = env.step(action)
            move_history.append(action_str)
            obs = result.observation
            
            if verbose:
                print(f"Reward: {result.reward:.2f}")
                print(obs.to_text())
        else:
            if verbose:
                print(f"Invalid action! Using random fallback.")
            if obs.legal_actions:
                import random
                fallback = random.choice(obs.legal_actions)
                action = ZipAction(fallback)
                result = env.step(action)
                move_history.append(f"({fallback})")
                obs = result.observation
    
    # Final result
    won = result.info.get("win", False)
    if verbose:
        print("\n" + "=" * 60)
        print(f"GAME OVER - {'WON!' if won else 'Lost'}")
        print(f"Moves: {move_history}")
        print(f"Solution was: {solution}")
        print("=" * 60)
    
    return won, move_history


if __name__ == "__main__":
    # Path to trained model (use the latest one)
    MODEL_PATH = "./zip_output/grpo-2026-01-30_19-21-26/final"
    
    # Load model
    model, tokenizer = load_trained_model(MODEL_PATH)
    
    # Create environment
    env = ZipEnv(size=6, num_count=8)
    
    # Play a game
    won, moves = play_game(model, tokenizer, env, max_turns=20, verbose=True)
    
    # Run multiple games for statistics
    print("\n" + "=" * 60)
    print("Running 10 games for statistics...")
    print("=" * 60)
    
    wins = 0
    for i in range(10):
        env_test = ZipEnv(size=6, num_count=8)
        w, _ = play_game(model, tokenizer, env_test, max_turns=20, verbose=False)
        if w:
            wins += 1
        print(f"Game {i+1}: {'Won' if w else 'Lost'}")
    
    print(f"\nWin rate: {wins}/10 ({wins*10}%)")
