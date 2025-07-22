from peft import PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import random
import numpy as np
import os

from utils.arg_parser import experts_merging_arg_parser
from merging_lora_modules.simple_averaging import SimpleAveraging
from merging_lora_modules.xlora_average import XLoraAveraging
from merging_lora_modules.arrow_routing import ArrowRouting
from data_handler.dataset import (
    apply_preprocessing,
    create_message_column_for_test
)
from utils.metrics import compute_generation_metrics
from utils.config import *
from typing import (
    List,
    Dict,
    Literal,
    Tuple

)
from huggingface_hub import hf_hub_download, login, list_repo_files
from MyConfig import *
from MyArgParser import downloading_adapters

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_clusters(repo_id: str, local_dir: str = "clusters"):
    os.makedirs(local_dir, exist_ok=True)

    # List all files in the repo
    all_files = list_repo_files(repo_id)

    for cluster_index in range(10):
        cluster_folder = f"cluster{cluster_index}"
        cluster_files = [f for f in all_files if f.startswith(f"{cluster_folder}/")]

        # Save path: clusters/cluster0, not clusters/cluster0/cluster0
        save_path = os.path.join(local_dir, cluster_folder)
        os.makedirs(save_path, exist_ok=True)

        print(f"Downloading files from Hugging Face subfolder '{cluster_folder}'...")

        for file_path in cluster_files:
            filename = file_path.split("/")[-1]  # get just the filename

            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=cluster_folder,
                local_dir=save_path
            )

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

def langauage_expert_adapters(repo_id: str, language_path: str,  local_dir: str="language_adapters") -> None :
    """
    This function will load the expert model
    :param repo_id: Repo ID.
    :param language_path: The path of language expert.
    :param local_dir: The folder we should save the cache.
    :return:
    Saving Wiki-adapters in local.
    """
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=language_path,
        local_dir=local_dir
    )

if __name__=='__main__':
    args = downloading_adapters()
    login(token=args.hf_token)
    load_all_clusters(CLUSTER_REPO_ID)
    langauage_expert_adapters(EXPERT_REPO_ID, LANGUAGE_PATH)
    model, tokenizer = model_and_tokenizer(model_name=MODEL_NAME)
