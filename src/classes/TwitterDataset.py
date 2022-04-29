import pandas as pd
import json
import torch
from torch.utils.data import Dataset


with open('./config/config.json', 'r') as f:
    config = json.load(f)


class TwitterDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_token_len=config['max_token_len']):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.LABEL_COLUMNS = list(self.data.columns)
        self.LABEL_COLUMNS.remove('text')
        self.TASK1_LABELS = self.LABEL_COLUMNS[:3]
        self.TASK2_LABELS = self.LABEL_COLUMNS[3:4]
        self.TASK3_LABELS = self.LABEL_COLUMNS[4:]

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        text = data_row.text
        labels1 = data_row[self.TASK1_LABELS]
        labels2 = data_row[self.TASK2_LABELS]
        labels3 = data_row[self.TASK3_LABELS]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels1=torch.FloatTensor(labels1),
            labels2=torch.FloatTensor(labels2),
            labels3=torch.FloatTensor(labels3)
        )

    def __len__(self):
        return len(self.data)
