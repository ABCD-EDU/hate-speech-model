import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from classes.TwitterDataModule import TwitterDataModule
from classes.TwitterNeuralNet import TwitterNeuralNet

with open('./config/config.json', 'r') as f:
    config = json.load(f)

print('-------READING TRAINING FILES-------')
task_df = pd.read_csv('../../res/preprocessed/task1_2/task1_2.csv')
task_df = task_df.dropna()
print('-------READING TRAINING FILES COMPLETED-------')
train_df = pd.read_csv('../../res/preprocessed/task1_2/train_task1_2.csv')
val_df = pd.read_csv('../../res/preprocessed/task1_2/val_task1_2.csv')
test_df = pd.read_csv('../../res/preprocessed/task1_2/test_task1_2.csv')
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

LABEL_COLUMNS = list(train_df.columns)
LABEL_COLUMNS.remove('text')

TASK1_LABELS = LABEL_COLUMNS[:2]
TASK2_LABELS = LABEL_COLUMNS[2:]

print('-------INITIALIZING TOKENIZER-------')
tokenizer = AutoTokenizer.from_pretrained(config['bert_model_name'])

print('-------CREATING DATA MODULE-------')
data_module = TwitterDataModule(
    train_df,
    val_df,
    test_df,
    tokenizer,
    batch_size=config["batch_size"],
    max_token_len=config["max_token_len"]
)
data_module.setup()
print('-------DATA MODULE CREATED-------')

steps_per_epoch = len(train_df) // config["batch_size"]
total_training_steps = steps_per_epoch * config["n_epochs"]
warmup_steps = total_training_steps // 5


logger = TensorBoardLogger("./logs/lightning_logs",
                           'twitter-task-all')

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

print('-------INITIALIZING TWITTER NEURAL NET-------')
model = TwitterNeuralNet(
    task1_n_classes=len(TASK1_LABELS),
    task2_n_classes=len(TASK2_LABELS),
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
)
print('-------NEURAL NET INITIALIZED-------')

print('-------TRAINING-------')

trainer = pl.Trainer(
    max_epochs=config['n_epochs'],
    gpus=1,
    progress_bar_refresh_rate=config['progress_bar_refresh_rate'],
    num_sanity_val_steps=config['num_sanity_val_steps'],
    fast_dev_run=False,
    logger=logger,
    checkpoint_callback=checkpoint_callback,
)
trainer.fit(model, data_module)

print('-------TRAINING COMPLETE-------')

print('-------EXPORTING MODEL-------')
PATH = './torch_model'
torch.save(model.state_dict(), os.path.join(PATH, "model.pt"))
print('-------MODEL EXPORTED SUCCESSFULLY-------')
