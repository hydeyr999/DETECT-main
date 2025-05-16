import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import random_split

def get_df(args,name_small=None,name_large=None):
    if args.mode == 'train':
        if args.task == 'otherdata':
            if args.run_logprobs == False:
                train_dir = f'./detectors/ling-based/logprobs/otherdata'
                paths = f'{train_dir}/{args.dataset}_{name_small}_{name_large}.json'
                df = pd.read_json(paths, lines=True)
            else:
                train_dir = f'./data/otherdata'
                df = pd.read_csv(f'{train_dir}/{args.dataset}_sample.csv')
        else:
            datasets_dict = {
                'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
                'cross-model': ['llama', 'deepseek', 'gpt4o','Qwen'],
                'cross-operation': ['create', 'translate','polish', 'expand','refine','summary','rewrite']}
            datasets = datasets_dict[args.task]
            if args.run_logprobs == False:
                train_dir = f'./detectors/ling-based/logprobs/{args.task}'
                paths = [f'{train_dir}/train/{x}_{name_small}_{name_large}.json' for x in datasets if x != args.dataset]
                print(paths)
                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_json(path, lines=True)

                    text = original['text'].values.tolist()
                    feat_small = original[f'logprob_small']
                    feat_large = original[f'logprob_large']
                    generated = original['generated'].values.tolist()

                    datas.extend(zip(text, feat_small, feat_large, generated))

                df = pd.DataFrame(datas, columns=['text', f'logprob_small', f'logprob_large', 'generated'])

                df = get_sample(df,args.n_sample)
            else:
                train_dir = f'./data/{args.task}'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                df = pd.DataFrame(datas, columns=['text', 'generated'])
                df = get_sample(df,args.n_sample)
    else:
        if args.run_logprobs == False:
            test_dir = f'./detectors/ling-based/logprobs/{args.task}'
            path = f'{test_dir}/test/{args.dataset}_{name_small}_{name_large}.json'
            df = pd.read_json(path, lines=True)
        else:
            if args.multilen > 0:
                test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
                df = pd.read_csv(test_dir)
                df['text'] = df[f'text_{args.multilen}']
            else:
                test_dir = f'./data/{args.task}'
                path = f'{test_dir}/{args.dataset}_sample.csv'
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    if len(df_human) == len(df_ai):
        try:
            data_df = pd.DataFrame({'text_human': df_human['text'],
                                    'text_ai': df_ai['text'],
                                    'logprob_small_human': df_human['logprob_small'],
                                    'logprob_small_ai': df_ai['logprob_small'],
                                    'logprob_large_human': df_human['logprob_large'],
                                    'logprob_large_ai': df_ai['logprob_large'],
                                    }).dropna().reset_index(drop=True)
        except:
            data_df = pd.DataFrame({'text_human': df_human['text'],
                                    'text_ai': df_ai['text'],
                                    }).dropna().reset_index(drop=True)
    else:
        df = df.sample(n=n_sample*2,random_state=12)
        return df

    data_df = data_df.sample(n=n_sample, random_state=12)
    print(data_df)

    text_human = data_df['text_human'].values.tolist()
    text_ai = data_df['text_ai'].values.tolist()
    text = text_human + text_ai
    generated = [0] * len(text_human) + [1] * len(text_ai)
    id = list(range(len(text)))
    try:
        logprob_small_human = data_df['logprob_small_human'].values.tolist()
        logprob_small_ai = data_df['logprob_small_ai'].values.tolist()
        logprob_large_human = data_df['logprob_large_human'].values.tolist()
        logprob_large_ai = data_df['logprob_large_ai'].values.tolist()
        logprob_small = logprob_small_human + logprob_small_ai
        logprob_large = logprob_large_human + logprob_large_ai
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'logprob_small': logprob_small,
                                  'logprob_large': logprob_large,
                                  'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'generated': generated})
    print(sample_df)

    return sample_df

def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_classifier(train_feats, train_labels):
    xgb = XGBClassifier(n_estimators=256, n_jobs=-1)
    lgb = LGBMClassifier(n_estimators=256, n_jobs=-1)
    cat = CatBoostClassifier(n_estimators=256, verbose=0)
    rfr = RandomForestClassifier(n_estimators=256, n_jobs=-1)
    model = VotingClassifier(
        n_jobs=-1,
        voting='soft',
        weights=[4, 5, 4, 4],
        estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat), ('rfr', rfr)]
    )
    model.fit(train_feats, train_labels)
    return model

def get_gltr_df(args,name_small=None, name_large=None):
    if args.mode == 'train':
        if args.task == 'otherdata':
            if args.run_feats == False:
                train_dir = f'./detectors/ling-based/lingfeatures/otherdata'
                paths = f'{train_dir}/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
                df = pd.read_json(paths, lines=True)
            else:
                train_dir = f'./data/otherdata'
                df = pd.read_csv(f'{train_dir}/{args.dataset}_sample.csv')
        else:
            datasets_dict = {
                'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
                'cross-model': ['llama', 'deepseek', 'gpt4o','Qwen'],
                'cross-operation': ['create', 'translate','polish', 'expand','refine','summary','rewrite']}
            datasets = datasets_dict[args.task]
            if args.run_feats == False:
                train_dir = f'./detectors/ling-based/lingfeatures/{args.task}'
                paths = [f'{train_dir}/train/{x}_gltrppl_{name_small}_{name_large}.json' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_json(path, lines=True)

                    text = original['text'].values.tolist()
                    gltr_feats = original['gltr_feats']
                    ppl_feats = original['ppl_feats']
                    generated = original['generated'].values.tolist()

                    datas.extend(zip(text, gltr_feats, ppl_feats, generated))

                df = pd.DataFrame(datas, columns=['text', 'gltr_feats', 'ppl_feats', 'generated'])

                df = get_ling_sample(df,args.n_sample)

            else:
                train_dir = f'./data/{args.task}'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                df = pd.DataFrame(datas, columns=['text', 'generated'])
                df = get_ling_sample(df,args.n_sample)
    else:
        if args.run_feats == False:
            test_dir = f'./detectors/ling-based/lingfeatures/{args.task}'
            path = f'{test_dir}/test/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
            df = pd.read_json(path, lines=True)
        else:
            if args.multilen > 0:
                test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
                df = pd.read_csv(test_dir)
                df['text'] = df[f'text_{args.multilen}']
            else:
                test_dir = f'./data/{args.task}'
                path = f'{test_dir}/{args.dataset}_sample.csv'
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_ling_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    try:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text'],
                                'gltr_feats_human': df_human['gltr_feats'],
                                'gltr_feats_ai': df_ai['gltr_feats'],
                                'ppl_feats_human': df_human['ppl_feats'],
                                'ppl_feats_ai': df_ai['ppl_feats'],
                                }).dropna().reset_index(drop=True)
    except:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text'],
                                }).dropna().reset_index(drop=True)
    data_df = data_df.sample(n=n_sample, random_state=12)
    print(data_df)

    text_human = data_df['text_human'].values.tolist()
    text_ai = data_df['text_ai'].values.tolist()
    text = text_human + text_ai
    generated = [0] * len(text_human) + [1] * len(text_ai)
    id = list(range(len(text)))
    try:
        gltr_feats_human = data_df['gltr_feats_human'].values.tolist()
        gltr_feats_ai = data_df['gltr_feats_ai'].values.tolist()
        ppl_feats_human = data_df['ppl_feats_human'].values.tolist()
        ppl_feats_ai = data_df['ppl_feats_ai'].values.tolist()
        gltr_feats = gltr_feats_human + gltr_feats_ai
        ppl_feats = ppl_feats_human + ppl_feats_ai
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'gltr_feats': gltr_feats,
                                  'ppl_feats': ppl_feats,
                                  'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'generated': generated})
    print(sample_df)

    return sample_df