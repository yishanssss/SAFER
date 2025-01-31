from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize


def compute_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    predicted = predicted.detach().cpu().numpy()
    return accuracy_score(y_true, predicted)



def compute_auc(y_pred, y_true, n_classes):
    # Ensure predictions are in probability form
    y_pred = torch.softmax(y_pred, dim=1).detach().cpu().numpy()

    # Binarize labels
    all_true_binary = np.zeros((y_true.size, n_classes))
    all_true_binary[np.arange(y_true.size), y_true] = 1

    # Compute Macro AUC (one-vs-rest approach)
    auc_macro_list = []
    for class_id in range(n_classes):
        true_binary = (y_true == class_id).astype(int)
        if len(np.unique((true_binary))) == 2:
            auc_macro_list.append(roc_auc_score(true_binary, y_pred[:, class_id]))
        else:
            continue
    auc_macro = np.mean(auc_macro_list)

    # Compute Micro AUC
    auc_micro = roc_auc_score(all_true_binary, y_pred, average="micro")

    return auc_macro, auc_micro

def HR_at_k(y_pred, y_true, k):
    y_pred = torch.softmax(y_pred, dim=1).detach().cpu()
    topk_preds = torch.topk(y_pred, k, dim=1).indices
    y_true = torch.tensor(y_true)
    y_true_expand = y_true.unsqueeze(1).repeat(1, k)
    hits = (topk_preds == y_true_expand).any(dim=1).float()
    hr_k = hits.mean().item()

    return hr_k

def MRR_at_k(y_pred, y_true, k):
    y_pred = torch.softmax(y_pred, dim=1).detach().cpu()
    topk_preds = torch.topk(y_pred, k, dim=1).indices
    y_true = torch.tensor(y_true)
    y_true_expand = y_true.unsqueeze(1).repeat(1, k)
    match_positions = (topk_preds == y_true_expand) * torch.arange(1, k+1).float().unsqueeze(0)
    reciprocal_ranks = 1.0 / match_positions[match_positions != 0]
    mrr_k = reciprocal_ranks.mean().item() if reciprocal_ranks.numel() > 0 else 0

    return mrr_k

def compute_fdr(y_pred, y_true, n_classes):
    y_pred_class = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
    
    fdr_list = []
    for class_id in range(n_classes):
        true_binary = (y_true == class_id).astype(int)
        pred_binary = (y_pred_class == class_id).astype(int)
        FP = sum(1 for true, pred in zip(true_binary, pred_binary) if true != pred)
        TP = sum(1 for true, pred in zip(true_binary, pred_binary) if true == pred)
        fdr = FP / (FP + TP)
        fdr_list.append(fdr)

    return np.mean(fdr_list)