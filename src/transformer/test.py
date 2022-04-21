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
#TODO: update TwitterNeuralNet params once trained with task3

loaded_model = TwitterNeuralNet(2,3) # Task1->2 labels | Task2->3 Labels
loaded_model.load_state_dict(torch.load(
    os.path.join(PATH, "model.pt"), map_location=device))

loaded_model = loaded_model.to(device)
loaded_model.eval()
loaded_model.freeze()

test_df = pd.read_csv('../../res/preprocessed/test_final.csv')
tokenizer = AutoTokenizer.from_pretrained(config['bert_model_name'])
test_dataset = TwitterDataset(test_df, tokenizer, max_token_len=config['max_token_len'])

def get_accuracy_score():
   task1_predictions = []
   task1_labels = []

   task2_predictions = []
   task2_labels = []

   with torch.no_grad():
      for idx, item in enumerate(test_dataset):
         _, prediction = loaded_model(
               item['input_ids'].unsqueeze(dim=0).to(device),
               item['attention_mask'].unsqueeze(dim=0).to(device)
         )

         task1_predictions.append(prediction[0].flatten())
         task2_predictions.append(prediction[1].flatten())
         
         task1_labels.append(item['labels1'].int())
         task2_labels.append(item['labels2'].int())
         

   task1_predictions_ = torch.stack(task1_predictions).detach().cpu()
   task1_labels_ = torch.stack(task1_labels).detach().cpu()

   task2_predictions_ = torch.stack(task2_predictions).detach().cpu()
   task2_labels_ = torch.stack(task2_labels).detach().cpu()

   THRESHOLD = 0.5

   task1_accuracy_score = accuracy(
      task1_predictions_, task1_labels_, threshold=THRESHOLD)

   task2_accuracy_score = accuracy(
      task2_predictions_, task2_labels_, threshold=THRESHOLD)

   print(f'task1_accuracy_score: {task1_accuracy_score}')
   print(f'task2_accuracy_score: {task2_accuracy_score}')

get_accuracy_score()
