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
    
    # --- MODIFICATION: ADD BATCH SIZE FOR GENERATION ---
    generation_batch_size: int = 4,  # Generate N variations at a time

    # --- Generation Parameters (Ranges for Variation) ---
    min_temperature: float = 0.6,
    max_temperature: float = 0.85,
    min_top_p: float = 0.8,
    max_top_p: float = 0.95,
    max_gen_len: int = 512,
    
    # --- System Configuration ---
    max_seq_len: int = 1024,
    # The `max_batch_size` is now determined by `generation_batch_size`
    seed: int = random.randint(1000, 2**31 - 1)
):
    """
    Generates multiple variations of a given MIDI file by prompting a trained model.

    This script loads a model once and then generates multiple song continuations from a
    single prompt file. For each generation, it uses a different, randomly selected
    temperature and top_p value from within the specified ranges, allowing for creative
    and diverse outputs from the same starting point.

    It processes generations in batches to better utilize GPU resources.
    """
    print("--- MIDI VARIATION GENERATION (BATCHED) ---")
    
    # --- VALIDATION ---
    if generation_batch_size > number_generations:
        print(f"Warning: generation_batch_size ({generation_batch_size}) is larger than number_generations ({number_generations}). Adjusting batch size to {number_generations}.")
        generation_batch_size = number_generations

    # --- 1. Setup seeds for reproducibility ---
    print(f"1. Setting random seed to {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --- 2. Load the model and tokenizer ---
    print(f"2. Loading the MusicLlama model with a max batch size of {generation_batch_size}... (This may take a moment)")
    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        # --- MODIFICATION: Use the new batch size for model setup ---
        max_batch_size=generation_batch_size,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
        seed=seed
    )
    print("    Model loaded successfully.")

    # --- 3. Prepare the prompt from a single file ---
    print(f"3. Preparing prompt from data in '{prompt_file_path}'...")
    try:
        # Load the MIDI data represented as a numpy array
        seed_data = np.load(prompt_file_path)
        # Encode the data and add a Start-Of-Sequence token
        seed_data_encoded = generator.tokenizer.encode_series(seed_data, if_add_sos=True, if_add_eos=False)
        total_tokens = len(seed_data_encoded)

        # Instead of taking the start, take a random slice from the song.
        if total_tokens <= prompt_len:
            print(f"    Warning: The song has only {total_tokens} tokens. Using the entire song as the prompt.")
            prompt_tokens = seed_data_encoded
        else:
            # Define the latest possible starting point for the slice
            max_start_index = total_tokens - prompt_len
            # Choose a random starting point
            start_index = random.randint(0, max_start_index)
            end_index = start_index + prompt_len
            prompt_tokens = seed_data_encoded[start_index:end_index]
            print(f"    Selected a random {prompt_len}-token segment starting at token {start_index}.")
            
    except FileNotFoundError:
        print(f"    ERROR: The prompt file '{prompt_file_path}' was not found.")
        return
    except Exception as e:
        print(f"    ERROR: Failed to load or process the prompt file. Details: {e}")
        return
        
    print(f"    Successfully created a prompt with {len(prompt_tokens)} tokens.")

    # --- 4. Generate music in a loop, creating variations ---
    # --- MODIFICATION: The loop now processes in batches ---
    print(f"\n4. Starting generation of {number_generations} variations in batches of {generation_batch_size}...")
    
    base_path, extension = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    
    generated_count = 0
    # Process in steps of `generation_batch_size`
    for i in range(0, number_generations, generation_batch_size):
        # Determine the size of the current batch (handles the last, possibly smaller, batch)
        current_batch_size = min(generation_batch_size, number_generations - i)
        
        # Prepare a batch of identical prompts
        batched_prompts = [prompt_tokens] * current_batch_size
        
        # Select random parameters for THIS BATCH
        temp = random.uniform(min_temperature, max_temperature)
        top_p_val = random.uniform(min_top_p, max_top_p)
        
        print(f"\n    (Batch {i//generation_batch_size + 1}) Generating {current_batch_size} variations with temp={temp:.3f}, top_p={top_p_val:.3f}...")

        results = generator.music_completion(
            batched_prompts,
            max_gen_len=max_gen_len,
            temperature=temp,
            top_p=top_p_val,
        )
        print("    ...Generation complete.")

        # --- 5. Save the generated MIDI files for this batch ---
        for j, result in enumerate(results):
            current_variation_index = i + j + 1
            current_output_path = f"{base_path}_{current_variation_index}_temp{temp:.2f}_topp{top_p_val:.2f}{extension}"
            result['generation']['content'].save(current_output_path)
            print(f"    ✅ Saved variation {current_variation_index} to: {current_output_path}")

        # Save the base prompt only once during the first batch
        if i == 0 and len(results) > 0:
            prompt_output_path = f"{base_path}_base_prompt{extension}"
            results[0]['generation']['prompt'].save(prompt_output_path)
            print(f"    ✅ Saved base prompt to: {prompt_output_path}")

    print("\n--- All variations are complete. ---")


if __name__ == "__main__":
    fire.Fire(generate_midi_variations)