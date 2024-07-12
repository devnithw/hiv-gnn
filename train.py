import torch 
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

import numpy as np
from tqdm import tqdm

from model import GNN
from dataset import HIVDataset


# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'

# Hyperparameters
BATCH_SIZE = 128
FEATURE_SIZE = 30


# Load datasets
print("Loading datasets...")
train_dataset = HIVDataset(root=TRAIN_DIR, filename=TRAIN_CSV)
print("Train dataset loaded.")
test_dataset = HIVDataset(root=TEST_DIR, filename=TEST_CSV)
print("Test dataset loaded.")

# Prepare Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataloaders ready.")

# Ensure feature size is 30
assert train_dataset[0].x.shape[1] == FEATURE_SIZE, "Feature size mismatch."

# Load model
model = GNN(feature_size=FEATURE_SIZE).to(device)
print(f"Model successfully loaded and sent to device: {device}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count number of trainable parameters
print(f"Number of trainable parameters: {n_params}")

# Loss function and optimizer
weights = torch.tensor([1, 8], dtype=torch.float32).to(device) # Class weights to handle class imbalance
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # LR decay

def train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device):
    """
    A function to train the model for one epoch and update weights
    """

    all_preds = []
    all_labels = []

    for _, batch in enumerate(tqdm(train_loader)):
        # Send data to device
        batch.to(device)

        # 1. Forward pass
        y_pred = model(batch.x.float(),
                       batch.edge_attr.float(),
                       batch.edge_index,
                       batch.batch)
    
        # 2. Calculate loss
        loss = torch.sqrt(loss_fn(y_pred, batch.y))


        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


