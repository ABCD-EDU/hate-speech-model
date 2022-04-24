import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from classes.TwitterNeuralNet import TwitterNeuralNet
from classes.TwitterDataset import TwitterDataset
from torchmetrics.functional import accuracy, f1_score, auroc


with open('./config/config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print('---------LOADING MODEL---------')
PATH = './torch_model'
# TODO: update TwitterNeuralNet params once trained with task3

loaded_model = TwitterNeuralNet(bert_model_name=config['bert_model_name'])  # Task1->2 labels | Task2->3 Labels
loaded_model.load_state_dict(torch.load(
    os.path.join(PATH, config["trained_model_name"]), map_location=device))
    # os.path.join(PATH, "bert_30_epoch.pt"), map_location=device))

loaded_model = loaded_model.to(device)
loaded_model.eval()
loaded_model.freeze()

test_df = pd.read_csv('../../res/preprocessed/test_final.csv')
tokenizer = AutoTokenizer.from_pretrained(config['bert_model_name'])
test_dataset = TwitterDataset(
    test_df, tokenizer, max_token_len=config['max_token_len'])

LABEL_COLUMNS = list(test_df.columns)
LABEL_COLUMNS.remove('text')

TASK1_LABELS = LABEL_COLUMNS[:3]
TASK2_LABELS = LABEL_COLUMNS[3:4]
TASK3_LABELS = LABEL_COLUMNS[4:]


task1_id2label = {idx: label for idx, label in enumerate(TASK1_LABELS)}
task1_label2id = {label: idx for idx, label in enumerate(TASK1_LABELS)}

task2_label2id = {label: idx for idx, label in enumerate(TASK2_LABELS)}
task2_id2label = {idx: label for idx, label in enumerate(TASK2_LABELS)}

task3_label2id = {label: idx for idx, label in enumerate(TASK3_LABELS)}
task3_id2label = {idx: label for idx, label in enumerate(TASK3_LABELS)}


def get_accuracy_score():
    task1_predictions = []
    task1_labels = []

    task2_predictions = []
    task2_labels = []

    task3_predictions = []
    task3_labels = []

    with torch.no_grad():
        for idx, item in enumerate(test_dataset):
            # print(test_df.iloc[idx].text)
            # if idx == 20:
            #     break
            _, prediction = loaded_model(
                item['input_ids'].unsqueeze(dim=0).to(device),
                item['attention_mask'].unsqueeze(dim=0).to(device)
            )

            task1_predictions.append(prediction[0].flatten())
            task2_predictions.append(prediction[1].flatten())
            task3_predictions.append(prediction[2].flatten())

            task1_labels.append(item['labels1'].int())
            task2_labels.append(item['labels2'].int())
            task3_labels.append(item['labels3'].int())

            # print('TASK 1')
            # print(prediction[0])
            # print(task1_id2label)
            # print(task1_id2label[int(torch.argmax(prediction[0]))])

            # print('TASK 2')
            # print(prediction[1])
            # print(task2_id2label)
            # print(task2_id2label[int(torch.argmax(prediction[1]))])

            # print('TASK 3')
            # print(prediction[2])
            # print(task3_id2label)
            # print(task3_id2label[int(torch.argmax(prediction[2]))])
            # print('=============================')

    task1_predictions_ = torch.stack(task1_predictions).detach().cpu()
    task1_labels_ = torch.stack(task1_labels).detach().cpu()
   #  print(task1_predictions)
   #  print(task2_predictions)
    task2_predictions_ = torch.stack(task2_predictions).detach().cpu()
    task2_labels_ = torch.stack(task2_labels).detach().cpu()

    task3_predictions_ = torch.stack(task3_predictions).detach().cpu()
    task3_labels_ = torch.stack(task3_labels).detach().cpu()

    THRESHOLD = 0.5

    task1_accuracy_score = accuracy(
        task1_predictions_, task1_labels_, threshold=THRESHOLD)

    task2_accuracy_score = accuracy(
        task2_predictions_, task2_labels_, threshold=THRESHOLD)

    task3_accuracy_score = accuracy(
        task3_predictions_, task3_labels_, threshold=THRESHOLD)

    print(f'task1_accuracy_score: {task1_accuracy_score}')
    print(f'task2_accuracy_score: {task2_accuracy_score}')
    print(f'task2_accuracy_score: {task3_accuracy_score}')


get_accuracy_score()
