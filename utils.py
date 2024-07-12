from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(y_pred, y_true, context="Train"):
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

    print(f"{context} - Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1 Score: {f1:.2f}")

def plot_loss_curve(train_losses, test_losses, output_path="loss_curve.png"):
    """
    A helper function to plot loss curves and export as an image.
    """

    plt.figure(figsize=(10, 6))
    # Calculate epoch
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot graphs
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    
    plt.savefig(output_path)
    plt.close()
    print(f'Loss curves saved as {output_path}')