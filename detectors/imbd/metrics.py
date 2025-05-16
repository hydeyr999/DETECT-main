import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np

def get_roc_metrics(real_preds, sample_preds):
    real_preds = [x for x in real_preds if not np.isnan(x)]
    sample_preds = [x for x in sample_preds if not np.isnan(x)]
    try:
        fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(e)
        return None
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    real_preds = [x for x in real_preds if not np.isnan(x)]
    sample_preds = [x for x in sample_preds if not np.isnan(x)]
    try:
        precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                      real_preds + sample_preds)
        pr_auc = auc(recall, precision)
    except Exception as e:
        print(e)
        return None
    return precision.tolist(), recall.tolist(), float(pr_auc)

