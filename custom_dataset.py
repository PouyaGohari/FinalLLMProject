from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets
from MyConfig import *

class CustomDataset(Dataset):
    def __init__(self, text_dataset:datasets, tokenizer:AutoTokenizer.from_pretrained, use_attention:bool=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = text_dataset
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.use_attention = use_attention

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
            return_dict=self.use_attention,
        )
        if self.use_attention:
            return {'input_ids': text['input_ids'].squeeze(0).to('cuda'), 'attention_mask': text['attention_mask'].squeeze(0).to('cuda')}
        return {'input_ids':text.squeeze(0).to('cuda')}
