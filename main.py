from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch

import random
import numpy as np
import os
import logging

from utils.config import *
from typing import (
    Tuple

)
from huggingface_hub import login, snapshot_download
from MyConfig import *
from MyArgParser import downloading_adapters

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def model_and_tokenizer(model_name: str, local_dir: str ="models") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    This function will load the quantized version of specified model.
    :param model_name: The model name
    :param local_dir: Directory of the downloaded the model
    :return:
        Model and Tokenizer.
    """
    os.makedirs(local_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side='right', model_max_length=MAX_LENGTH, cache_dir=local_dir
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, quantization_config=bnb_config, cache_dir=local_dir
    )
    return model, tokenizer

def load_save_hf_repo(repo_id: str,  local_dir: str="language_adapters") -> None :
    """
    This function will load the expert model
    :param repo_id: Repo ID.
    :param local_dir: The folder we should save the cache.
    :return:
    Saving Wiki-adapters in local.
    """
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir
    )
    print(f"Downloaded the entire repo {repo_id} to {local_dir}/")

if __name__=='__main__':
    args = downloading_adapters()
    login(token=args.hf_token)
    load_save_hf_repo(CLUSTER_REPO_ID, local_dir="clusters")
    load_save_hf_repo(EXPERT_REPO_ID, local_dir="language_adapters")
    model, tokenizer = model_and_tokenizer(model_name=MODEL_NAME)
