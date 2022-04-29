from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import json

with open('./config/config.json', 'r') as f:
    config = json.load(f)


class SingleTaskNN(pl.LightningModule):
    def __init__(self, task_name, taskn_classes,  n_training_steps=None, n_warmup_steps=None, bert_model_name=None):
        super().__init__()

        self.task_name = task_name
        self.bert = AutoModel.from_pretrained(
            bert_model_name, return_dict=True)

        self.hidden = nn.Linear(self.bert.config.hidden_size,
                                self.bert.config.hidden_size)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, taskn_classes)

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.criterion_CE = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_BCE = nn.BCELoss(reduction="mean")

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(pooled_output)

        loss = 0

        output = self.classifier(pooled_output)
        if self.task_name == "task1":
            output = F.softmax(output, dim=1)
            if labels is not None:
                loss = self.criterion_CE(output, labels['labels1'])
                # print(labels['labels1'])
        # IF TASK 2 or 3
        else:
            output = torch.sigmoid(output)
            if labels is not None:
                if self.task_name == "task2":
                    loss = self.criterion_BCE(output, labels['labels2'])
                else:
                    loss = self.criterion_BCE(output, labels['labels3'])


        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = {}
        labels['labels1'] = batch["labels1"]
        labels['labels2'] = batch["labels2"]
        labels['labels3'] = batch["labels3"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = {}
        labels['labels1'] = batch["labels1"]
        labels['labels2'] = batch["labels2"]
        labels['labels3'] = batch["labels3"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = {}
        labels['labels1'] = batch["labels1"]
        labels['labels2'] = batch["labels2"]
        labels['labels3'] = batch["labels3"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=config['learning_rate'], weight_decay=config['w_decay'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
