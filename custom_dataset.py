from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets
from MyConfig import *

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
            padding='max_length',
            truncation=True,
            tokenize=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {'input_ids': text.squeeze(0).to('cuda')}
