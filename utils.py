from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def calculate_metrics(y_pred, y_true):
    """
    A helper function to calculate the metrics for a given set of predictions and true labels. The calculated metrics are:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    Prints to the screen the calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1 Score: {f1:.2f}")