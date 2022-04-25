import os
import pandas as pd
import json
import torch
from transformers import AutoTokenizer
from classes.TwitterNeuralNet import TwitterNeuralNet

with open('./config/config.json', 'r') as f:
    config = json.load(f)

print('-------IMPORTING MODEL-------')
PATH = './torch_model'
torch.set_printoptions(precision=10)
torch.set_printoptions(sci_mode=False)
# torch.set_printoptions(precision=6)

device = torch.device('cpu')
# Task1->2 labels | Task2->3 Labels
loaded_model = TwitterNeuralNet(bert_model_name=config['bert_model_name'])
loaded_model.load_state_dict(torch.load(
    os.path.join(PATH, config["trained_model_name"]), map_location=device))

loaded_model.eval()
loaded_model.freeze()


task_df = pd.read_csv('../../res/preprocessed/train_final.csv')
LABEL_COLUMNS = list(task_df.columns)
LABEL_COLUMNS.remove('text')

TASK1_LABELS = LABEL_COLUMNS[:3]
TASK2_LABELS = LABEL_COLUMNS[3:5]
TASK3_LABELS = LABEL_COLUMNS[5:]


task1_id2label = {idx: label for idx, label in enumerate(TASK1_LABELS)}
task1_label2id = {label: idx for idx, label in enumerate(TASK1_LABELS)}

task2_label2id = {label: idx for idx, label in enumerate(TASK2_LABELS)}
task2_id2label = {idx: label for idx, label in enumerate(TASK2_LABELS)}

task3_label2id = {label: idx for idx, label in enumerate(TASK3_LABELS)}
task3_id2label = {idx: label for idx, label in enumerate(TASK3_LABELS)}


print('-------INITIALIZING TOKENIZER-------')
tokenizer = AutoTokenizer.from_pretrained(config['bert_model_name'])
while True:
    input_text = input('Input a sentence: ')
    encoded_text = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=config['max_token_len'],
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        test_input_ids, test_att_mask = encoded_text['input_ids'], encoded_text['attention_mask']
        _, output = loaded_model(test_input_ids, test_att_mask)

    print('TASK 1')
    print(output[0])
    print(task1_id2label)
    print(task1_id2label[int(torch.argmax(output[0]))])

    print('TASK 2')
    print(output[1])
    print(task2_id2label)
    print(task2_id2label[int(torch.argmax(output[1]))])

    print('TASK 3')
    print(output[2])
    print(task3_id2label)
    print(task3_id2label[int(torch.argmax(output[2]))])

    prompt = input('Do you wish to continue? Y|N: ')
    if prompt.lower() == 'y':
        continue
    else:
        break
