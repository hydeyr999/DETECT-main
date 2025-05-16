import argparse
import pandas as pd
from deberta import *
from data import *
import torch

def train_deberta(args,train_df):
    model_path = './models/deberta-v3-large'
    model_weights = get_deberta_train(model_path, train_df,device = args.device,num_epoch=args.epochs,threshold = args.threshold)
    bert_save_dir = f'./detectors/deberta/weights/{args.task}'
    torch.save(model_weights, f'{bert_save_dir}/deberta_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}')

def get_deberta_test(args,test_df):
    model_path = './models/deberta-v3-large'
    model_weights = args.deberta_model
    bert_save_dir = f'./detectors/deberta/results/{args.task}'
    bert_save_path = f'{bert_save_dir}/deberta_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}.csv'
    get_bert_result(args,model_path,model_weights,test_df,bert_save_path,'deberta')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--deberta_model',type=str,default=None)
    parser.add_argument('--mode', type=str,default='test',choices=['train','test'])
    parser.add_argument('--device', type=str,default='cuda:0')
    parser.add_argument('--epochs', type=int,default=4)
    parser.add_argument('--threshold', type=float,default=0)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--n_sample',type=int,default=2000)
    args = parser.parse_args()

    if args.mode == 'train':
        train_df = get_df(args).dropna().reset_index(drop=True).sample(n=20)
        print(train_df)
        train_deberta(args,train_df)
    elif args.mode == 'test':
        test_df = get_df(args).dropna().reset_index(drop=True).sample(n=20)
        get_deberta_test(args,test_df)
    else:
        raise NotImplementedError