import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
from data import *
import tqdm
import torch
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)

def run_mpu(args):
    tokenizer = AutoTokenizer.from_pretrained("./detectors/opensource/mpu_env2")
    detector = AutoModelForSequenceClassification.from_pretrained("./detectors/opensource/mpu_env2").to(args.device)

    df = get_df(args).dropna().reset_index(drop=True)
    texts = df['text'].values.tolist()
    generated = df['generated'].values.tolist()

    preds = []
    labels = []
    for idx in tqdm.tqdm(range(len(texts))):
        text = texts[idx]
        label = generated[idx]
        try:
            tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
            with torch.no_grad():
                pred = detector(**tokenized).logits.softmax(-1)[0, 1].item()
            preds.append(pred)
            labels.append(label)
        except Exception as e:
            continue

    _,_,auc_score = get_roc_metrics(labels, preds)
    print(f'AUC Score: {auc_score}')
    import pandas as pd
    results_df = pd.DataFrame({'auc': [auc_score]})
    results_df.to_csv(f'./detectors/opensource/results/{args.task}/mpu_{args.dataset}_multi{args.multilen}.csv',index=False)


def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['cross-domain', 'cross-operation', 'cross-model'],default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--max_len',type=int,default=512)
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()

    run_mpu(args)