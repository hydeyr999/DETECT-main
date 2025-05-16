import os

from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import tqdm
import torch

class CustomDataset(Dataset):
    def __init__(self, data_json_dir):
        with open(data_json_dir, 'r') as f:
            data_json = json.load(f)
        self.data = self.process_data(data_json)

    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        original_text = self.data['original'][index]
        sampled_text = self.data['rewritten'][index]

        return {
            'text': [original_text, sampled_text],
            'label': [0, 1]  # Original label is 0, Sampled label is 1
        }

    def process_data(self, data_json):
        processed_data = {
            'original': data_json['original'],
            'rewritten': data_json['rewritten']
        }

        return processed_data

class CustomDataset_rewrite(Dataset):
    def __init__(self, args,data_json_dir,mode='train'):
        if mode == 'train':
            if args.task_name == "ai_detection_500":
                self.data_json_dir = './detectors/imbd/ai_detection_500_polish.raw_data.json'
                with open(self.data_json_dir, 'r') as f:
                    data_df = json.load(f)
                self.data = self.process_data(data_df)
            else:
                datasets_dict = {
                    'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
                    'cross-model': ['llama', 'deepseek', 'gpt4o','Qwen'],
                    'cross-operation': ['create', 'translate', 'polish','expand','refine','summary','rewrite']}
                datasets = datasets_dict[args.task_name]

                train_dir = f'./data/{args.task_name}'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != data_json_dir]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                data_df = pd.DataFrame(datas, columns=['text', 'generated'])

                data_df_human = data_df[data_df['generated']==0].reset_index(drop=True)
                data_df_ai = data_df[data_df['generated']==1].reset_index(drop=True)
                assert len(data_df_human) == len(data_df_ai)
                data_df = pd.DataFrame({'original': data_df_human['text'], 'rewritten': data_df_ai['text']}).dropna().reset_index(drop=True)
                data_df = data_df.sample(n=2000, random_state=12)
                print(data_df)

                self.data = self.process_data(data_df)

        else:
            if args.cut_len>0:
                test_dir = f'./data/multilen/len_{args.cut_len}/{args.task_name}/{args.eval_dataset}_len_{args.cut_len}.csv'
                data_df = pd.read_csv(test_dir)
                data_df['text'] = data_df[f'text_{args.cut_len}']
                print(data_df)
            else:
                if data_json_dir in ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets']:
                    task_name = 'cross-domain'
                elif data_json_dir in ['llama', 'deepseek', 'gpt4o','Qwen']:
                    task_name = 'cross-model'
                elif data_json_dir in ['create', 'translate', 'polish','expand','refine','summary','rewrite']:
                    task_name = 'cross-operation'
                else:
                    self.data_json_dir = data_json_dir
                    with open(data_json_dir, 'r') as f:
                        data_df = json.load(f)
                    self.data = self.process_data(data_df)
                    return

                test_dir = f'./data/{task_name}'
                path = f'{test_dir}/{data_json_dir}_sample.csv'
                data_df = pd.read_csv(path)

            data_df_human = data_df[data_df['generated'] == 0].reset_index(drop=True)
            data_df_ai = data_df[data_df['generated'] == 1].reset_index(drop=True)
            assert len(data_df_human) == len(data_df_ai)
            data_df = pd.DataFrame({'original': data_df_human['text'], 'rewritten': data_df_ai['text']}).dropna().reset_index(drop=True)
            print(data_df)

            self.data = self.process_data(data_df)


    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        original_text = self.data['original'][index]
        rewritten_text = self.data['rewritten'][index]

        return original_text, rewritten_text

    def process_data(self, data_df):
        try:
            processed_data = {
                'original': data_df['original'].tolist(),
                'rewritten': data_df['rewritten'].tolist()
            }
        except Exception as e:
            processed_data = {
                'original': data_df['original'],
                'rewritten': data_df['rewritten']
            }
        return processed_data

class CustomDataset_split(Dataset):
    def __init__(self, data_json_dir, split='train', val_ratio=0.2):
        with open(data_json_dir, 'r') as f:
            data_json = json.load(f)
        self.data = self.process_data(data_json)
        
        total_size = len(self.data['original'])
        
        if val_ratio == 0: 
            self.indices = [i for i in range(total_size)]
            return
            
        # Compute step size for stratified sampling
        step_size = int(1 / val_ratio)

        val_indices = list(range(0, total_size, step_size))
        train_indices = [i for i in range(total_size) if i not in val_indices]
        # print(val_indices)
        # print(train_indices)
        if split == 'train':
            self.indices = train_indices
        elif split == 'val':
            self.indices = val_indices
        else:
            raise ValueError("split must be either 'train' or 'val'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        original_text = self.data['original'][actual_index]
        sampled_text = self.data['rewritten'][actual_index]
        return original_text, sampled_text

    def process_data(self, data_json):
        processed_data = {
            'original': data_json['original'],
            'rewritten': data_json['rewritten']
        }

        return processed_data
