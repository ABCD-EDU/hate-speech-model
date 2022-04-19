from torch.utils.data import DataLoader
from  classes.TwitterDataset import TwitterDataset
import json
import pytorch_lightning as pl

with open('./config/config.json', 'r') as f:
    config = json.load(f)

num_workers = config['num_workers']

class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=config['batch_size'], max_token_len=config['max_token_len']):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = TwitterDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )
        self.val_dataset = TwitterDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = TwitterDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
