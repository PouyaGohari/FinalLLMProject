from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import datasets
from utils.config import *


class CustomDataset(Dataset):
    def __init__(self, text_dataset:datasets, tokenizer:AutoTokenizer.from_pretrained):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = text_dataset
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user = {
            "content":f"{self.dataset[idx]}",
            "role":"user"
        }
        text = self.tokenizer.apply_chat_template(
            [user],
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            tokenize=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        return {"input_ids":input_ids, 'attention_mask':attention_mask}
