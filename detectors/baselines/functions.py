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

def get_df(args):
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


def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)
