import pandas as pd
import numpy as np
import torch
import gc
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AIDataset(Dataset):
    def __init__(self, text_df, tokenizer, max_len):
        self.text_list_ori=text_df['text'].values
        self.text_list_gen=text_df['generated_text'].values
        self.tokenizer=tokenizer
        self.max_len=max_len
        try:
            self.label_list = text_df['generated'].values
        except Exception:
            pass
    def __len__(self):
        return len(self.text_list_ori)
    def __getitem__(self, index):
        try:
            text_ori = self.text_list_ori[index]
            tokenized_ori = self.tokenizer(text=text_ori,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            text_gen = self.text_list_gen[index]
            tokenized_gen = self.tokenizer(text=text_gen,
                                           padding='max_length',
                                           truncation=True,
                                           max_length=self.max_len,
                                           return_tensors='pt')
            try:
                label = self.label_list[index]
                return (tokenized_ori['input_ids'].squeeze(),tokenized_gen['input_ids'].squeeze(), tokenized_ori['attention_mask'].squeeze(), label)
            except Exception:
                return (tokenized_ori['input_ids'].squeeze(),tokenized_gen['input_ids'].squeeze(), tokenized_ori['attention_mask'].squeeze())
        except Exception as e:
            print(f"Skipping sample at index {index} due to error: {e}")
            return None

class DPICModel(nn.Module):
    def __init__(self, model_path, config, tokenizer, pretrained=False,num_classes=1):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_path, config=config)
        else:
            self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(2*(config.hidden_size), num_classes)

    def get_embeddings(self,input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings = sum_embeddings / sum_mask
        # embeddings = outputs.last_hidden_state[:,0,:]

        gc.collect()
        torch.cuda.empty_cache()
        del outputs

        return embeddings
    def forward_features(self, input_ids_ori,input_ids_gen, attention_mask=None):
        embeddings_ori = self.get_embeddings(input_ids_ori, attention_mask)
        embeddings_gen = self.get_embeddings(input_ids_gen, attention_mask)
        embeddings_concat = torch.cat((embeddings_ori, embeddings_gen), dim=1)
        return embeddings_concat

    def forward(self, input_ids_ori,input_ids_gen, attention_mask):
        embeddings_concat = self.forward_features(input_ids_ori,input_ids_gen, attention_mask)
        logits = self.classifier(embeddings_concat)

        return logits
