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
from huggingface_hub import hf_hub_download, login, list_repo_files
import MyConfig

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_clusters(repo_id: str, local_dir: str = "clusters"):
    os.makedirs(local_dir, exist_ok=True)

    # List all files in the repo
    all_files = list_repo_files(repo_id)

    # Loop through cluster0 to cluster9
    for cluster_index in range(10):
        cluster_folder = f"cluster{cluster_index}"

        # Filter files belonging to this subfolder
        cluster_files = [f for f in all_files if f.startswith(f"{cluster_folder}/")]

        print(f"Downloading from: {cluster_folder} ({len(cluster_files)} files)")

        for file in cluster_files:
            relative_path = file.split("/", 1)[1]  # get filename only
            save_path = os.path.join(local_dir, cluster_folder)
            os.makedirs(save_path, exist_ok=True)

            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=relative_path,
                subfolder=cluster_folder,
                local_dir=save_path,
                cache_dir=None,
                local_dir_use_symlinks=False
            )
            print(f"  ✓ Downloaded: {local_file}")


if __name__=='__main__':
    hf_token = input("Please give your token to logging purpose\n")
    login(token=hf_token)
    load_all_clusters(MyConfig.cluster_repo_id)
