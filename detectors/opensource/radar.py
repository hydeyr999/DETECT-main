import transformers
import torch
import argparse
from data import get_df
import tqdm
import torch.nn.functional as F
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)

def get_preds(args,detector,tokenizer,text):
    with torch.no_grad():
      inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
      inputs = {k:v.to(args.device) for k,v in inputs.items()}
      output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
    return output_probs[0]

def get_detect(args):
    detector = transformers.AutoModelForSequenceClassification.from_pretrained("./detectors/opensource/radar-vicuna-7b")
    tokenizer = transformers.AutoTokenizer.from_pretrained("./detectors/opensource/radar-vicuna-7b")
    detector.eval()
    detector.to(args.device)

    df = get_df(args).dropna().reset_index(drop=True)
    texts = df['text'].values.tolist()
    preds=[]
    for text in tqdm.tqdm(texts):
        pred = get_preds(args,detector,tokenizer,text)
        preds.append(pred)

    _,_,auc = get_roc_metrics(df['generated'].values.tolist(),preds)
    _,_,pr_auc = get_precision_recall_metrics(df['generated'].values.tolist(),preds)
    print("AUC: ",auc,"PR AUC: ",pr_auc)
    import pandas as pd
    results_df = pd.DataFrame({'auc':[auc],'pr_auc':[pr_auc]})
    results_df.to_csv(f'./detectors/opensource/results/{args.task}/radar_{args.dataset}_multi{args.multilen}.csv', index=False)

def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default=None)
    parser.add_argument('--dataset',type=str,default=None)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--device',type=str,default='cuda:2')
    args = parser.parse_args()
    print(args)

    get_detect(args)
