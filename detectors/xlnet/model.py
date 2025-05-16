from transformers import XLNetModel, XLNetConfig,XLNetTokenizer
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel,get_linear_schedule_with_warmup, AutoModelForCausalLM,BitsAndBytesConfig
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch import optim
import tqdm
import gc
import pandas as pd
import numpy as np

class XLNet_Model(nn.Module):
    def __init__(self, model_path, config, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = XLNetModel.from_pretrained(model_path, config=config)
        else:
            self.model = XLNetModel._from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward_features(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        return pooled_output

    def forward(self, input_ids, attention_mask, token_type_ids):
        embeddings = self.forward_features(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(embeddings)
        return logits

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

class XLNETDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len, label_list=None):
        self.text_list=text_list
        self.tokenizer=tokenizer
        self.max_len=max_len
        try:
            self.label_list = label_list
        except Exception:
            pass
    def __len__(self):
        return len(self.text_list)
    def __getitem__(self, index):
        try:
            text = self.text_list[index]

            tokenized = self.tokenizer(text=text,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')

            try:
                label = self.label_list[index]
                return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(),label
            except Exception:
                return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()
        except Exception as e:
            print(f"Skipping sample at index {index} due to error: {e}")
            return None


def get_eval(probs_df, df ,model_name):
    roc_auc = roc_auc_score(df.generated.values, probs_df.generated.values)
    print(f'{model_name}_roc_auc:', roc_auc)


def get_xlnet_train(model_path,df,threshold=0,learning_rate = 0.00001,max_len = 512,batch_size = 4,num_epoch = 10,device='cuda:0'):
    config = XLNetConfig.from_pretrained(model_path)
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    model = XLNet_Model(model_path, config, pretrained=True)
    model.to(device)
    print('xlnet model loaded.')
    print('device:', device)

    train_texts = df['text'].values
    train_labels = df['generated'].values

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_train_steps = int(len(train_texts) / batch_size * num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    train_dataset = XLNETDataset(train_texts, tokenizer, max_len, train_labels)
    train_generator = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=False)
    print('training ready.')

    scaler = GradScaler()
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (input_ids, attention_mask, token_type_ids,label) in enumerate(train_generator):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.float().to(device)
            token_type_ids = token_type_ids.to(device)

            with autocast():
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), label)

            losses.update(loss.item(), input_ids.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            print('\r', end='', flush=True)
            message = '%s %10.4f %6.1f    |     %0.3f     |' % ("train", j / len(train_generator) + ep, ep, losses.avg)
            print(message, end='', flush=True)

            if threshold > 0:
                if losses.avg < threshold:
                    print("stopping early")
                    print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)
                    return model.state_dict()


        print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    return model.state_dict()

def get_prediction(model_path, model_weights, df,device='cuda:0'):
    text_list = df['text'].values

    max_len = 512
    batch_size = 16

    config = XLNetConfig.from_pretrained(model_path)
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    model = XLNet_Model(model_path, config, pretrained=False)
    model.load_state_dict(torch.load(model_weights,map_location={'cuda:0':'cuda:1'}))
    model.to(device)
    model.eval()

    test_datagen = XLNETDataset(text_list, tokenizer, max_len)
    test_generator = DataLoader(dataset=test_datagen,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)
    pred_prob = np.zeros((len(text_list),), dtype=np.float32)
    for j, (batch_input_ids, batch_attention_mask,batch_token_type_ids) in tqdm.tqdm(enumerate(test_generator), total=len(test_generator)):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(test_generator) - 1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            pred_prob[start:end] = logits.sigmoid().cpu().data.numpy().squeeze()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pred_prob

def get_xlnet_result(args,model_path, model_weights, df, output_dir, model_name):
    xlnet_prob0 = get_prediction(model_path, model_weights, df,device=args.device)
    gc.collect()
    torch.cuda.empty_cache()

    xlnet_probs_df = pd.DataFrame(data={'id': df.id.values, 'generated': xlnet_prob0})
    xlnet_probs_df.to_csv(output_dir, index=False)
    xlnet_probs_df = pd.read_csv(output_dir)
    print('xlnet_probs_df:', xlnet_probs_df)

    get_eval(xlnet_probs_df, df, model_name)

    return xlnet_probs_df
