import numpy as np
from sklearn.metrics import (precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix)


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, probabilities, threshold):
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, predictions)),
                precision=float(precision_score(y, predictions, zero_division=0)),
                recall=float(recall_score(y, predictions, zero_division=0)),
                f1=float(f1_score(y, predictions, zero_division=0)),
                roc_auc=float(roc_auc_score(y, probabilities)),
                average_precision=float(average_precision_score(y, probabilities)),
                brier_score=float(brier_score_loss(y, probabilities)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
