from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import gcc
import os
import logging
import random
import numpy as np

from typing import (
    Tuple
)
from huggingface_hub import login, snapshot_download
from MyConfig import *
from MyArgParser import arg_parser
from my_cka import (
    load_general_dataset,
    apply_arrow_or_gks,
    get_samples, create_torch_dataset,
    dataloader,
    apply_cka
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_and_tokenizer(model_name: str, local_dir: str ="models") -> Tuple[AutoModelForCausalLM.from_pretrained, AutoTokenizer.from_pretrained]:
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

if __name__=='__main__':
    args = arg_parser()
    set_seed(args.seed)
    login(token=args.hf_token)

    general_model, tokenizer = model_and_tokenizer(model_name=MODEL_NAME)

    dataset = load_general_dataset(path=args.dataset_path, data_file=DATA_FILE)
    sub_dataset = get_samples(your_dataset=dataset['test'], n_samples=args.n_samples, seed=args.seed)

    print(f"------------- Subsampling from {args.dataset_path} has been finished and enhanced model is starting to be processed------------------------")

    enhanced_model = apply_arrow_or_gks(
        base_model_name=args.base_model_name,
        cluster_names=list(CLUSTER_NAMES.values()),
        arrow_top_k=args.top_k,
        arrow_router_temperature=args.temperature,
        gks=args.gks,
        language_experts=list(LANGUAGE_EXPERTS.values()),
    )


    my_generator = torch.Generator()
    my_generator.manual_seed(args.seed)
    print(sub_dataset[0])

    compatible_dataset = create_torch_dataset(sub_dataset, tokenizer)
    print(compatible_dataset[0])

    my_dataloader = dataloader(
        compatible_dataset=compatible_dataset,
        batch=args.batch,
        generator=my_generator
    )


    print(f"------------- Starting to apply cka -------------")

    if args.gks:
        second_model_name = "GenKowlSub"
    else:
        second_model_name = "Arrow"
    layers_of_interest = [f"model.layers.{i}.self_attn.o_proj" for i in range(31)] + [f"model.layers.{i}.self_attn.qkv_proj" for i in range(31)]
    result = []
    for layer in layers_of_interest:
        exported_data = apply_cka(
            first_loader=my_dataloader,
            base_model=general_model,
            base_model_layers=layer,
            enhanced_model_layers=layers_of_interest,
            enhanced_model=enhanced_model,
            first_model_name="Baseline",
            second_model_name=second_model_name,
            export_data=args.export_data,
            show_plot=args.show_plot,
            device=args.device
        )
        gcc.collect()
        torch.cuda.empty_cache()
        result.append(exported_data)
