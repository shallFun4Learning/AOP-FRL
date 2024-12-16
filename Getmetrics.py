import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix,average_precision_score
import torch
from torch import nn, scalar_tensor

def getMetrics(eval_preds):
    logits, y_true = eval_preds
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)

    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)

    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp/(tp+fn)

    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba[:, 1])
    AUC = auc(FPR, TPR)
    return {'ACC': ACC, 'MCC': MCC, 'Recall': Sn, 'AUC': AUC}


def getScore(logits):
    # Convert logits to tensor if it is not already a tensor
    if not torch.is_tensor(logits):
        logits = torch.tensor(logits)
    # Reshape logits to 2D if it is 1D
    if logits.ndimension() == 1:
        logits = logits.unsqueeze(0)
    # Apply softmax to logits
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    return y_proba




def getPredictLabel(logits):
    # Convert logits to tensor if it is not already a tensor
    if not torch.is_tensor(logits):
        logits = torch.tensor(logits)
    # Reshape logits to 2D if it is 1D
    if logits.ndimension() == 1:
        logits = logits.unsqueeze(0)
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)
    return y_pred
