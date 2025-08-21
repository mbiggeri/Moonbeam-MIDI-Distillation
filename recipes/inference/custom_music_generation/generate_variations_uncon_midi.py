# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import fire
import torch
import random
import os
import numpy as np
from typing import Optional

# Import the unconditional MusicLlama class
from generation import MusicLlama

def generate_midi_variations(
    # --- Model and Tokenizer Paths ---
    ckpt_dir: str,
    model_config_path: str,
    tokenizer_path: str,
    finetuned_PEFT_weight_path: Optional[str] = None,

    # --- Data and Prompting ---
    prompt_file_path: str = "processed/prompt_song.npy",
    prompt_len: int = 50,

    # --- Generation Control ---
    number_generations: int = 20,
    output_path: str = "variations_output/variation.mid",
    
    # --- Generation Parameters (Ranges for Variation) ---
    min_temperature: float = 0.6,
    max_temperature: float = 0.85,
    min_top_p: float = 0.8,
    max_top_p: float = 0.95,
    max_gen_len: int = 512,
    
    # --- System and Performance Configuration ---
    batch_size: int = 8,
    max_seq_len: int = 1024,
    seed: int = random.randint(1000, 2**31 - 1)
):
    """
    Generates multiple variations of a given MIDI file using batch processing for higher GPU utilization.
    """
    print("--- MIDI VARIATION GENERATION (GPU OPTIMIZED) ---")

    # --- 1. Setup seeds for reproducibility ---
    print(f"1. Setting random seed to {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --- 2. Load the model and tokenizer ---
    print(f"2. Loading the MusicLlama model with a batch size of {batch_size}...")
    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
        seed=seed
    )
    print("   Model loaded successfully.")

    # --- 3. Prepare the prompt from a single file ---
    print(f"3. Preparing prompt from data in '{prompt_file_path}'...")
    try:
        seed_data = np.load(prompt_file_path)
        seed_data_encoded = generator.tokenizer.encode_series(seed_data, if_add_sos=True, if_add_eos=False)
        total_tokens = len(seed_data_encoded)
        if total_tokens <= prompt_len:
            print(f"   Warning: The song has only {total_tokens} tokens. Using the entire song as the prompt.")
            prompt_tokens = seed_data_encoded
        else:
            max_start_index = total_tokens - prompt_len
            start_index = random.randint(0, max_start_index)
            end_index = start_index + prompt_len
            prompt_tokens = seed_data_encoded[start_index:end_index]
            print(f"   Selected a random {prompt_len}-token segment starting at token {start_index}.")
    except FileNotFoundError:
        print(f"   ERROR: The prompt file '{prompt_file_path}' was not found.")
        return
    except Exception as e:
        print(f"   ERROR: Failed to load or process the prompt file. Details: {e}")
        return
    print(f"   Successfully created a prompt with {len(prompt_tokens)} tokens.")

    # --- 4. Generate music in batches ---
    print(f"\n4. Starting generation of {number_generations} variations in batches of {batch_size}...")
    
    base_path, extension = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    for i in range(0, number_generations, batch_size):
        current_batch_size = min(batch_size, number_generations - i)
        temp = random.uniform(min_temperature, max_temperature)
        top_p_val = random.uniform(min_top_p, max_top_p)
        
        print(f"\n   Generating batch starting at song {i+1} (size: {current_batch_size}) with temp={temp:.3f}, top_p={top_p_val:.3f}...")
        
        prompts_batch = [prompt_tokens] * current_batch_size

        results = generator.music_completion(
            prompts_batch,
            max_gen_len=max_gen_len,
            temperature=temp,
            top_p=top_p_val,
        )
        print("   ...Batch generation complete.")

        # --- CORRECTION START ---
        # Save the prompt, but only on the very first batch (i == 0).
        # We get the savable prompt object from the 'results' dictionary.
        if i == 0:
            prompt_output_path = f"{base_path}_base_prompt{extension}"
            # The 'prompt' is the same for all items in the batch, so we take it from the first result.
            results[0]['generation']['prompt'].save(prompt_output_path)
            print(f"   ✅ Saved base prompt to: {prompt_output_path}")
        # --- CORRECTION END ---

        # --- 5. Save the generated MIDI files for this batch ---
        for j, result in enumerate(results):
            file_index = i + j + 1
            current_output_path = f"{base_path}_{file_index}_temp{temp:.2f}_topp{top_p_val:.2f}{extension}"
            result['generation']['content'].save(current_output_path)
            print(f"   ✅ Saved variation to: {current_output_path}")

    print("\n--- All variations are complete. ---")


if __name__ == "__main__":
    fire.Fire(generate_midi_variations)