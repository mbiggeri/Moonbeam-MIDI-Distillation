# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
import random
import numpy as np
import torch
from dataclasses import asdict

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.utils.dataset_utils import DATASET_PREPROC


def set_seed(seed):
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    if train_config.peft_method not in names:
        raise RuntimeError(f"Peft config not found: {train_config.peft_method}")

    if train_config.peft_method == "prefix":
        raise RuntimeError("PrefixTuning is currently not supported (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089350811)")

    if train_config.enable_fsdp and train_config.peft_method == "llama_adapter":
        raise RuntimeError("Llama_adapter is currently not supported in combination with FSDP (see https://github.com/meta-llama/llama-recipes/issues/359#issuecomment-2089274425)")

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params = asdict(config)
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()

    update_config(dataset_config, **kwargs)

    return  dataset_config


def get_distillation_configs(
    # Aggiungi qui gli argomenti specifici della distillazione con valori di default
    teacher_model_name: str="meta-llama/Llama-2-7b-hf",
    alpha: float=0.5,
    temperature: float=2.0,
    **kwargs
):
    from llama_recipes.configs import train_config as TRAIN_CONFIG
    from llama_recipes.configs import model_config as MODEL_CONFIG

    # Aggiorna train_config con i parametri specifici della distillazione
    TRAIN_CONFIG.teacher_model_name = teacher_model_name
    TRAIN_CONFIG.alpha = alpha
    TRAIN_CONFIG.temperature = temperature
    
    # Aggiorna le altre configurazioni come nella funzione `get_train_configs`
    # (Questa è una versione semplificata, puoi estenderla se necessario)
    model_configs = {k:v for k,v in MODEL_CONFIG.__dict__.items()}
    
    return model_configs, TRAIN_CONFIG, None, None, None


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        if train_config.batching_strategy == "padding":
            if train_config.enable_fsdp:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=mode=="train",
                )
            else:
                kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            if train_config.enable_fsdp:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
                drop_last=True,
            )
            elif train_config.enable_ddp:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
                drop_last=True,
                )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

        return kwargs