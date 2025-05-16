import argparse
import pandas as pd
from model import *
from data import *
import torch

def train_xlnet(args,train_df):
    model_path = './models/xlnet-large'
    model_weights = get_xlnet_train(model_path, train_df,device = args.device,num_epoch=args.epochs,threshold = args.threshold)
    xlnet_save_dir = f'./detectors/xlnet/weights/{args.task}'
    torch.save(model_weights, f'{xlnet_save_dir}/xlnet_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}')

def get_xlnet_test(args,test_df):
    model_path = './models/xlnet-large'
    model_weights = args.xlnet_model
    xlnet_save_dir = f'./detectors/xlnet/results/{args.task}'
    xlnet_save_path = f'{xlnet_save_dir}/xlnet_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}.csv'
    get_xlnet_result(args,model_path,model_weights,test_df,xlnet_save_path,'xlnet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--xlnet_model',type=str)
    parser.add_argument('--mode', type=str,default='test',choices=['train','test'])
    parser.add_argument('--device', type=str,default='cuda:0')
    parser.add_argument('--epochs', type=int,default=4)
    parser.add_argument('--threshold', type=float,default=0)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--n_sample', type=int, default=2000)
    args = parser.parse_args()

    if args.mode == 'train':
        train_df = get_df(args).dropna().reset_index(drop=True)
        print(train_df)
        train_xlnet(args,train_df)
    elif args.mode == 'test':
        test_df = get_df(args).dropna().reset_index(drop=True)
        get_xlnet_test(args,test_df)


