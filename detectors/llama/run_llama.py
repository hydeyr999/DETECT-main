import argparse
from data import *
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import pandas as pd

def get_eval(probs_df, df ,model_name):
    roc_auc = roc_auc_score(df.generated.values, probs_df.generated.values)
    print(f'{model_name}_roc_auc:', roc_auc)

def get_llm_result(args,test_cfg,save_dir,model_id,df,model_name):
    import run_llm_inference

    run_llm_inference.main(test_cfg,save_dir, model_id,df,device = args.device)
    llm_probs_df_m0 = pd.read_parquet(f'{save_dir}/{model_id}.parquet')

    llm_probs_df_m0 = llm_probs_df_m0.sort_values(by='id')
    print(llm_probs_df_m0)

    get_eval(llm_probs_df_m0,df,model_name)

    return llm_probs_df_m0

def train_llm(args,train_df):
    import llama3
    llama_dir = './detectors/llama'
    if args.imbddata:
        train_cfg = OmegaConf.load(f'{llama_dir}/conf/otherdata/train/conf_llama_imbd.yaml')
    else:
        train_cfg = OmegaConf.load(f'{llama_dir}/conf/train.yaml')
        train_cfg.train_params.num_train_epochs = args.epochs
        train_cfg.outputs.model_dir = f'{llama_dir}/weights/{args.task}/llama_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}'
    print(args.device)
    print(train_cfg)
    llama3.run_training(train_cfg,train_df,args.device,args.threshold)

def get_llm_test(args,test_df):
    llama_dir = './detectors/llama'
    save_dir = f'{llama_dir}/results/{args.task}'
    if args.imbddata:
        test_cfg = OmegaConf.load(f'{llama_dir}/conf/otherdata/test/conf_llama_imbd.yaml')
        save_name = f'llama_{args.dataset}_imbd'
    else:
        test_cfg = OmegaConf.load(f'{llama_dir}/conf/test.yaml')
        test_cfg.model.lora_path = f'{llama_dir}/weights/{args.task}/llama_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}/best'
        save_name = f'llama_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}'
    print(args.device)
    print(test_cfg)

    get_llm_result(args,test_cfg,save_dir,save_name,test_df,'llama')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train','test'],default='train')
    parser.add_argument('--threshold', type=float,default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wo_domain', type=str, default=None)
    parser.add_argument('--n_sample',type=int,default=2000)
    parser.add_argument('--multilen', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--imbddata', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        if args.task == 'otherdata':
            train_dir = './data/otherdata'
            train_df = pd.read_csv(f'{train_dir}/{args.dataset}_sample.csv')
            print(train_df)
        else:
            train_df = get_df(args).dropna()
            print(train_df)
        train_llm(args,train_df)
    elif args.mode == 'test':
        print(args.dataset)
        test_df = get_df(args).dropna().reset_index(drop=True)
        test_df['id'] = test_df.index
        print(test_df)
        if args.wo_domain:
            test_df = test_df[test_df['domain']!=args.wo_domain].reset_index(drop=True)
            print(test_df)
        get_llm_test(args,test_df)
