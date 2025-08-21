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

def generate_unconditional_music(
    # --- Model and Tokenizer Paths ---
    ckpt_dir: str,
    model_config_path: str,
    tokenizer_path: str,
    finetuned_PEFT_weight_path: Optional[str] = None,

    # --- Data and Prompting ---
    data_dir: str = "processed/",
    prompt_len: int = 5,

    # --- Generation Control ---
    number_generations: int = 1,
    output_path: str = "unconditional_output/generated.mid",
    
    # --- Generation Parameters ---
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_gen_len: int = 512,
    
    # --- System Configuration ---
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    seed: int = random.randint(1000, 2**31 - 1)  # Random seed for reproducibility,
    # seed: int = 1867000932  # Fixed seed for reproducibility, can be changed as needed
):
    """
    Generates one or more unconditional MIDI files from a trained model.

    This script prompts the model with short, random snippets from a given data directory
    and lets it generate the continuation.
    """
    print("--- UNCONDITIONAL MIDI GENERATION ---")

    # --- 1. Setup seeds for reproducibility ---
    print(f"1. Setting random seed to {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --- 2. Load the model and tokenizer ---
    print("2. Loading the MusicLlama model...")
    generator = MusicLlama.build(
        ckpt_dir=ckpt_dir,
        model_config_path=model_config_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path=finetuned_PEFT_weight_path,
        seed=seed
    )
    print("   Model loaded successfully.")

    # --- 3. Prepare the prompts ---
    print(f"3. Preparing {number_generations} prompts from data in '{data_dir}'...")
    try:
        # Find all processed numpy files to use as seeds
        seed_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if not seed_files:
            print(f"   ERROR: No '.npy' files found in the directory '{data_dir}'. Cannot create prompts.")
            return

        if len(seed_files) < number_generations:
            print(f"   Warning: Requested {number_generations} generations, but only found {len(seed_files)} seed files. Using duplicates.")
            seed_files_sampled = random.choices(seed_files, k=number_generations)
        else:
            seed_files_sampled = random.sample(seed_files, number_generations)

    except FileNotFoundError:
        print(f"   ERROR: The data directory '{data_dir}' was not found.")
        return

    prompts = []
    for filename in seed_files_sampled:
        # Load the MIDI data represented as numpy array
        seed_data = np.load(os.path.join(data_dir, filename))
        # Encode the data and add a Start-Of-Sequence token
        seed_data_encoded = generator.tokenizer.encode_series(seed_data, if_add_sos=True, if_add_eos=False)
        # Use the first few tokens as the prompt
        prompts.append(seed_data_encoded[:prompt_len])
    
    print(f"   Successfully created {len(prompts)} prompts, each with {prompt_len} tokens.")

    # --- 4. Generate music in a single batch ---
    print(f"\n4. Calling the model to generate {len(prompts)} sequences...")
    results = generator.music_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print("   Generation complete.")

    # --- 5. Save the generated MIDI files ---
    print("\n5. Saving output files...")
    base_path, extension = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    for i, result in enumerate(results):
        current_output_path = f"{base_path}_{i}{extension}" if number_generations > 1 else output_path
        prompt_output_path = f"{base_path}_{i}_prompt{extension}" if number_generations > 1 else f"{base_path}_prompt{extension}"

        # Save the generated music
        result['generation']['content'].save(current_output_path)
        
        # Save the short prompt that was used to generate the music
        result['generation']['prompt'].save(prompt_output_path)
        
        print(f"   ✅ Saved generation to: {current_output_path}")
        print(f"   ✅ Saved prompt to:    {prompt_output_path}")

    print("\n--- All generations are complete. ---")


if __name__ == "__main__":
    fire.Fire(generate_unconditional_music)