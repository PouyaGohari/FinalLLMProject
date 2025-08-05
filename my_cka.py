from torch_cka import CKA
import torch
from torch.utils.data import DataLoader
import datasets
from custom_dataset import CustomDataset
from tqdm import tqdm
from warnings import warn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
##Note that this is peft from the written in Guidance file
from peft import ArrowConfig, create_arrow_model, PeftModel

from typing import (
    Dict,
    List,
    Optional
)

class CustomCKA(CKA):
    def __init__(self,
        base_model:AutoModelForCausalLM,
        enhanced_model:PeftModel,
        first_model_name:str,
        second_model_name:str,
        base_model_layers: Optional[List[str]] = None,
        enhanced_model_layers: Optional[List[str]] = None,
        device:str='cuda'):
        super().__init__(
            model1=base_model,
            model2=enhanced_model,
            model1_name=first_model_name,
            model2_name=second_model_name,
            model1_layers=base_model_layers,
            model2_layers=enhanced_model_layers,
            device=device
        )

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1, dtype=K.dtype, device=self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for x1, x2 in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.squeeze(1).to(self.device))
            _ = self.model2(x2.squeeze(1).to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mismatch! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches
            del x1, x2, _, self.model1_features, self.model2_features

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())
        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

def apply_arrow_or_gks(
        base_model_name:str,
        cluster_names:List[str],
        arrow_top_k:int,
        arrow_router_temperature:float,
        gks:Optional[bool],
        language_experts:Optional[List[str]],
   ) -> PeftModel:
    """
    This function will either apply arrow routing mechanism to adapters or general knowledge subtraction in conjunction with arrow router based on the gks parameter.
    :param base_model_name: The model name or path for containing safetensors.
    :param cluster_names: A dictionary for clusters(tasks-specific lora-adapters) where keys is the adapter names and values are the paths for corresponding adapters.
    :param arrow_top_k: The top k of loras for routing among.
    :param arrow_router_temperature: The temperature that applies to softmax of arrow.
    :param gks: If applying general knowledge subtraction.
    :param language_experts: A dictionary where keys are the name of adapters(e.g, English or French) and values contains the paths for corresponding adapters.
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
    arrow_config = ArrowConfig(
        arrow_top_k = arrow_top_k,
        arrow_router_temperature = arrow_router_temperature,
        use_gks = gks,
    )
    model = create_arrow_model(
        base_model = base_model,
        task_specific_adapter_paths = cluster_names,
        general_adapter_paths = language_experts,
        arrow_config = arrow_config,
    )
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
    cka = CustomCKA(
        base_model,
        enhanced_model,
        first_model_name,
        second_model_name,
        base_model_layers,
        enhanced_model_layers,
        device
    )

    cka.compare(
        dataloader1=first_loader,
        dataloader2=second_loader,
    )

    if show_plot:
        cka.plot_results()
    if export_data:
        return cka.export()
    return None