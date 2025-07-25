# Add the parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from local_peft.tuners.lora.layer import LoraLayer
from merging_lora_modules.base_merging_module import BaseMergingModule, cluster_checkpoint_names
from peft import PeftModel
import torch


class CrossLingualExpertOrganiser(BaseMergingModule):
    def __init__(
            self, base_model, tokenizer, model_name,
            source_formal_expert_path, target_formal_expert_path,
            method, alpha=0.5, beta=0.5, load_single_expert=False, cluster_name=None
    ):
        super().__init__(base_model, tokenizer, model_name)

        self.source_formal_expert_path = source_formal_expert_path
        self.target_formal_expert_path = target_formal_expert_path
        self.method = method
        # a and b are coefficients for functional and target formal expert for being combined
        self.alpha = alpha
        self.beta = beta

        if load_single_expert:
            assert cluster_name, 'Cluster idx should be passed'
            self.base_model = PeftModel.from_pretrained(
                self.base_model, cluster_checkpoint_names[cluster_name], adapter_name=cluster_name)
            self.cluster_names = {cluster_name: cluster_checkpoint_names[cluster_name]}
        else:
            self.load_lora_modules()
            self.cluster_names = cluster_checkpoint_names

    def average_lora_modules(self):
        adapter_names = list(self.cluster_names.keys())
        self.base_model.load_adapter(self.cluster_names['cluster0'], adapter_name="average")

        for module in self.base_model.modules():
            if isinstance(module, LoraLayer):
                first_adapter = adapter_names[0]
                sum_A = torch.zeros_like(module.lora_A[first_adapter].weight)
                sum_B = torch.zeros_like(module.lora_B[first_adapter].weight)

                for adapter_name in adapter_names:
                    sum_A += module.lora_A[adapter_name].weight
                    sum_B += module.lora_B[adapter_name].weight

                avg_A = sum_A / len(adapter_names)
                avg_B = sum_B / len(adapter_names)

                module.lora_A["average"].weight = torch.nn.Parameter(avg_A)
                module.lora_B["average"].weight = torch.nn.Parameter(avg_B)

    def create_functional_modules(self, use_avg_lora):
        self.base_model.load_adapter(self.source_formal_expert_path, adapter_name='source_formal_expert')
        
        if self.method == 'subtract':
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in self.cluster_names.keys():
                        if use_avg_lora:
                            ## I guess it is for subtracting each lora module from the averaged lora experts
                            module.lora_A[adapter_name].weight = torch.nn.Parameter(
                                module.lora_A[adapter_name].weight - module.lora_A['average'].weight)
                            module.lora_B[adapter_name].weight = torch.nn.Parameter(
                                module.lora_B[adapter_name].weight - module.lora_B['average'].weight)
                        else:
                            ## I guess it is for subtracting the Wikipedia
                            module.lora_A[adapter_name].weight = torch.nn.Parameter(
                                module.lora_A[adapter_name].weight - module.lora_A['source_formal_expert'].weight)
                            module.lora_B[adapter_name].weight = torch.nn.Parameter(
                                module.lora_B[adapter_name].weight - module.lora_B['source_formal_expert'].weight)

        elif self.method == 'orthogonal_projection':
            for module in self.base_model.modules():
                if isinstance(module, LoraLayer):
                    if use_avg_lora:
                        Q_A, _ = torch.linalg.qr(module.lora_A['average'].weight.T)
                        Q_B, _ = torch.linalg.qr(module.lora_B['average'].weight)
                    else:
                        Q_A, _ = torch.linalg.qr(module.lora_A['source_formal_expert'].weight.T)
                        Q_B, _ = torch.linalg.qr(module.lora_B['source_formal_expert'].weight)

                    for adapter_name in self.cluster_names.keys():
                        mixed_expert_A = module.lora_A[adapter_name].weight.T
                        mixed_expert_B = module.lora_B[adapter_name].weight

                        projection_coefficients_A = Q_A.T @ mixed_expert_A  
                        projection_coefficients_B = Q_B.T @ mixed_expert_B  

                        mixed_project_on_formal_A = Q_A @ projection_coefficients_A  
                        mixed_project_on_formal_B = Q_B @ projection_coefficients_B

                        module.lora_A[adapter_name].weight = torch.nn.Parameter(mixed_expert_A.T - mixed_project_on_formal_A.T)
                        module.lora_B[adapter_name].weight = torch.nn.Parameter(mixed_expert_B - mixed_project_on_formal_B)

        self.base_model.delete_adapter("source_formal_expert")
        print("Functional Expert has been created successfully.")
    
    def create_cross_lingual_expert(self):
        self.base_model.load_adapter(self.target_formal_expert_path, adapter_name='target_formal_expert')

        for module in self.base_model.modules():
            if isinstance(module, LoraLayer):
                for adapter_name in self.cluster_names.keys():
                    module.lora_A[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_A[adapter_name].weight + self.beta * module.lora_A['target_formal_expert'].weight)
                    module.lora_B[adapter_name].weight = torch.nn.Parameter(self.alpha * module.lora_B[adapter_name].weight + self.beta * module.lora_B['target_formal_expert'].weight)

        self.base_model.delete_adapter("target_formal_expert")
        print("Cross Lingual Expert has been created successfully.")

    def merge(self, add_functional_only, use_avg_lora=False):
        if use_avg_lora:
            self.average_lora_modules()

        if add_functional_only:
            self.create_functional_modules(use_avg_lora)
        else:
            self.create_functional_modules(use_avg_lora)
            self.create_cross_lingual_expert()

        if use_avg_lora:
            self.base_model.delete_adapter('average')
