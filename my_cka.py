from torch_cka import CKA
import torch
from torch.utils.data import DataLoader
import datasets
from custom_dataset import CustomDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
##Note that this is peft from the written in Guidance file
from peft import PeftModel, LoraConfig

from typing import (
    Dict,
    List,
    Optional
)

def apply_arrow_or_gks(
        base_model_name:str,
        cluster_names:Dict,
        arrow_top_k:int,
        arrow_router_temperature:float,
        gks:bool=False,
        language_experts:Dict=None,
        target_modules:List[str] = None
   ) -> PeftModel:
    """
    This function will either apply arrow routing mechanism to adapters or general knowledge subtraction in conjunction with arrow router based on the gks parameter.
    :param base_model_name: The model name or path for containing safetensors.
    :param cluster_names: A dictionary for clusters(tasks-specific lora-adapters) where keys is the adapter names and values are the paths for corresponding adapters.
    :param arrow_top_k: The top k of loras for routing among.
    :param arrow_router_temperature: The temperature that applies to softmax of arrow.
    :param gks: If applying general knowledge subtraction.
    :param language_experts: A dictionary where keys are the name of adapters(e.g, English or French) and values contains the paths for corresponding adapters.
    :param target_modules: The modules where adapters are being target.
    :return:
    A Peft Model where all adapters applied with respect to the method.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, quantization_config=bnb_config)
    model = PeftModel.from_pretrained(
        base_model, cluster_names['cluster0'], adapter_name="cluster0"
    )
    for cluster_name, cluster_dir in cluster_names.items():
        if cluster_name != "cluster0":
            model.load_adapter(cluster_dir, adapter_name=cluster_name)
    if gks:
        for language_name_expert, language_expert_dir in language_experts.items():
            model.load_adapter(language_expert_dir, adapter_name=language_name_expert)
        arrow_config = LoraConfig(
            r=2,  # dummy rank since A and B won't be used!
            use_arrow=True,  # This will turn this LoRA to the ArrowLoraVariant
            arrow_expert_num=len(cluster_names),  # Number of task-specific modules in each LoRA layer
            arrow_top_k=arrow_top_k,  # Number of selected task-specific LoRAs for each token in each layer
            arrow_router_temperature=arrow_router_temperature,
            use_gks=gks,  # ← enable GenKnowSub!
            le_names=list(language_experts.keys()),  # name of loaded general-domain LoRAs
            ts_names=list(cluster_names.keys()),  # name of loaded task-specific LoRAs
            target_modules=target_modules
        )
    else:
        arrow_config = LoraConfig(
            r=2,  # dummy rank since A and B won't be used!
            use_arrow=True,  # This will turn this LoRA to the ArrowLoraVariant
            arrow_expert_num=len(cluster_names),  # Number of task-specific modules in each LoRA layer
            arrow_top_k=arrow_top_k,  # Number of selected task-specific LoRAs for each token in each layer
            arrow_router_temperature=arrow_router_temperature,
            use_gks=gks,  # ← enable GenKnowSub!
            ts_names=list(cluster_names.keys()),  # name of loaded task-specific LoRAs
            target_modules=target_modules
        )
    model.add_adapter(adapter_name="router", peft_config=arrow_config)
    model.set_adapter("router")
    return model

def load_general_dataset(path:str, data_file:Dict) -> datasets:
    """
    This function gets a path and data file to return an existing dataset in your disk.
    :param path: The path of dataset.
    :param data_file: The train and test files.
    :return:
    """
    return datasets.load_dataset(path=path, data_files=data_file)

def get_samples(your_dataset:datasets, n_samples:int, seed:int=42) -> datasets:
    """
    This function will return a subset of a dataset.
    :param your_dataset: Your dataset.
    :param n_samples: Number of samples you need from your dataset.
    :param seed: For reproducibility.
    :return:
    Subset of corresponding dataset.
    """
    return your_dataset.shuffle(seed).select(range(n_samples))

def create_torch_dataset(dataset:datasets, tokenizer:AutoTokenizer.from_pretrained) -> CustomDataset:
    """
    This function will create custom dataset compatible with torch dataset.
    :param dataset: The text dataset(it must be for testing purpose.)
    :param tokenizer: The tokenizer to tokenize each input text.
    :return:
    CustomDataset.
    """
    return CustomDataset(
        text_dataset=dataset,
        tokenizer=tokenizer
    )

def dataloader(compatible_dataset:CustomDataset, generator:torch.Generator, batch:int=8, shuffle:bool=True) -> DataLoader:
    """
    This function will create a data loader.
    :param compatible_dataset: A custom dataset instantiated form class above.
    :param batch: The number of samples in each batch.
    :param shuffle: Shuffling dataset before batching.
    :param generator: The generator for reproducibility
    :return:
    DataLoader.
    """
    return DataLoader(compatible_dataset, batch_size=batch, shuffle=shuffle, generator=generator)

def apply_cka(
        first_loader:DataLoader,
        base_model:AutoModelForCausalLM,
        enhanced_model:PeftModel,
        first_model_name:str,
        second_model_name:str,
        export_data:Optional[bool] = False,
        show_plot:Optional[bool] = False,
        base_model_layers: Optional[List[str]] = None,
        enhanced_model_layers: Optional[List[str]] = None,
        second_loader: Optional[DataLoader] = None,
        device:str='cuda',
    ) -> Optional[Dict]:
    """
    This function will compare two given models and two different dataset as you wish just like documentation of the torch_cka library.
    :param first_loader: The data loader for first dataset.
    :param base_model: The first model in our case base model.
    :param enhanced_model:  The configured model like general knowledge subtracted model or arrow itsel model.
    :param first_model_name: The name of first model.
    :param second_model_name: The name of second model.
    :param export_data: If you want to export the data after comparing two models.
    :param show_plot: If you want to plot the data after comparing two models.
    :param base_model_layers: The specified layer of the base model.
    :param enhanced_model_layers: The specified layer of the second model.
    :param second_loader: If you have two dataset, pass the second loader correspond to that.
    :param device: If you want to use cuda or cpu.
    :return:
    If export_data has been set to true it would return a dictionary contain the data after comparing two models otherwise None would be returned.
    """
    cka = CKA(
        model1=base_model,
        model2=enhanced_model,
        model1_name=first_model_name,
        model2_name=second_model_name,
        model1_layers=base_model_layers,
        model2_layers=enhanced_model_layers,
        device=device
    )

    print("Collected model1 features:", cka.model1_features.keys())
    print("Collected model2 features:", cka.model2_features.keys())

    cka.compare(
        dataloader1=first_loader,
        dataloader2=second_loader,
    )
    if show_plot:
        cka.plot_results()
    if export_data:
        return cka.export()
    return None