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
    Literal
)
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import MyConfig

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loadClusterFile(repo_id: str, filenames: list[str], subfolder: str = "", local_dir: str = "clusters") -> None:
    """
    Load each cluster file from Hugging Face repo into local environment.

    :param repo_id: The ID of the Hugging Face repository (e.g. "username/repo_name")
    :param filenames: A list of cluster filenames to download
    :param subfolder: Optional subfolder in the repo (if any)
    :param local_dir: Local directory to save downloaded files
    """
    os.makedirs(local_dir, exist_ok=True)

    for filename in filenames:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir
        )
        print(f"Loaded: {file_path}")


if __name__=='__main__':
    hf_token = input("Please give your token to logging purpose")
    login(token=hf_token)
    loadClusterFile(MyConfig.cluster_repo_id, MyConfig.file_names)
