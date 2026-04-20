"""Evaluation metrics for NanoTorch."""

import numpy as np
from Tensor import Tensor

def accuracy(output, target):
    """
    Computes accuracy for classification.
    Args:
        output: Tensor of logits or probabilities (batch, num_classes)
        target: Tensor of target indices (batch,)
    """
    if isinstance(output, Tensor):
        output = output.data
    if isinstance(target, Tensor):
        target = target.data
    
    preds = np.argmax(output, axis=1)
    return np.mean(preds == target)

def precision_recall_f1(output, target, average='macro'):
    """
    Computes precision, recall, and F1-score.
    Args:
        output: Tensor of logits or probabilities (batch, num_classes)
        target: Tensor of target indices (batch,)
        average: 'macro' or 'micro'
    """
    if isinstance(output, Tensor):
        output = output.data
    if isinstance(target, Tensor):
        target = target.data
        
    preds = np.argmax(output, axis=1)
    num_classes = output.shape[1]
    
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        tp = np.sum((preds == c) & (target == c))
        fp = np.sum((preds == c) & (target != c))
        fn = np.sum((preds != c) & (target == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    if average == 'macro':
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    else:
        # Micro (equivalent to accuracy for multi-class classification)
        total_tp = np.sum(preds == target)
        total_fp_fn = np.sum(preds != target)
        micro_score = total_tp / (total_tp + total_fp_fn)
        return micro_score, micro_score, micro_score

def confusion_matrix(output, target, num_classes=None):
    """
    Computes a confusion matrix.
    """
    if isinstance(output, Tensor):
        output = output.data
    if isinstance(target, Tensor):
        target = target.data
        
    preds = np.argmax(output, axis=1)
    if num_classes is None:
        num_classes = output.shape[1]
        
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(target, preds):
        cm[t, p] += 1
    return cm
