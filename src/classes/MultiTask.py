from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import json

with open('./config/config.json', 'r') as f:
    config = json.load(f)


class MultiTaskNN(pl.LightningModule):
    def __init__(self, task1_n_classes: int = 3, task2_n_classes: int = 1, task3_n_classes: int = 5,  n_training_steps=None, n_warmup_steps=None, bert_model_name=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model_name, return_dict=True)
        self.hidden = nn.Linear(self.bert.config.hidden_size,
                                self.bert.config.hidden_size)

        self.task1_classifier = nn.Linear(
            self.bert.config.hidden_size, task1_n_classes)

        self.task2_classifier = nn.Linear(
            self.bert.config.hidden_size, task2_n_classes)

        self.task3_classifier = nn.Linear(
            self.bert.config.hidden_size, task3_n_classes)

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.task1_classifier.weight)
        torch.nn.init.xavier_uniform_(self.task2_classifier.weight)
        torch.nn.init.xavier_uniform_(self.task3_classifier.weight)

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_BCE = nn.BCELoss(reduction="mean")
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(pooled_output)

        output1 = self.task1_classifier(pooled_output)
        output2 = self.task2_classifier(pooled_output)
        output3 = self.task3_classifier(pooled_output)

        output1 = F.softmax(output1, dim=1)
        output2 = torch.sigmoid(output2)
        output3 = torch.sigmoid(output3)

        loss = 0
        if labels is not None:

            loss1 = self.criterion_CE(output1, labels['labels1'])
            loss2 = self.criterion_BCE(output2, labels['labels2'])
            loss3 = self.criterion_BCE(output3, labels['labels3'])
            # loss = torch.mean(loss1*.33+loss2*.33+loss3*.33)
            loss = loss1+loss2+loss3

        return loss, [output1, output2, output3]

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
