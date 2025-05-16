import generate
from model import *
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import tqdm
from data import *
from torch.nn import DataParallel


def get_dpic_train(model,tokenizer,df,device='cuda:0',batch_size=4,num_epoch=3,learning_rate = 0.00001,max_len = 512):
    model.to(device)
    print('deberta model loaded.')
    print('device:', device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_train_steps = int(len(df) / batch_size * num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    train_dataset = AIDataset(df, tokenizer, max_len)
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
        for j, (input_ids_ori, input_ids_gen, attention_mask, label) in enumerate(train_generator):
            input_ids_ori = input_ids_ori.to(device)
            input_ids_gen = input_ids_gen.to(device)
            attention_mask = attention_mask.to(device)
            label = label.float().to(device)

            with autocast():
                logits = model(input_ids_ori, input_ids_gen, attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), label)

            losses.update(loss.item(), input_ids_ori.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            print('\r', end='', flush=True)
            message = '%s %10.4f %6.1f    |     %0.3f     |' % (
            "train", j / len(train_generator) + ep, ep, losses.avg)
            print(message, end='', flush=True)

        print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    return model.state_dict()

def save_model(args,model):
    model_weights = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
    save_path = os.path.join(args.model_save_dir, f'{args.task}/dpic_{args.dataset}_ep{args.epochs}_n{args.n_sample}.pth')
    torch.save(model_weights, save_path)
    print(f"Model saved to {save_path}")

def run_dpic_train(args):
    if args.run_generate:
        df = generate.run(args)
        df.to_json(f'{args.data_save_dir}/{args.task}/train/{args.dataset}_sample.json', orient='records', lines=True)
    else:
        df = get_dpic_df(args).dropna().reset_index(drop=True)
        print(df)


    model_path = args.backbone
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    deberta_model = DPICModel(model_path,config,tokenizer)

    model_weights = get_dpic_train(deberta_model,
                                   tokenizer,
                                   df,
                                   batch_size=args.batch_size,
                                   max_len=args.max_len,
                                   device=args.device,
                                   num_epoch=args.epochs)

    model_name = os.path.basename(model_path)
    save_model(args, deberta_model)



