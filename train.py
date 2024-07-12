import torch 
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

import numpy as np
from tqdm import tqdm

from model import GNN
from dataset import HIVDataset
from utils import calculate_metrics, plot_loss_curve


# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
MODEL_SAVE_PATH = "GNN_model.pth"

# Hyperparameters
BATCH_SIZE = 128
FEATURE_SIZE = 30
EPOCHS = 1


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

def train_step(model, train_loader, loss_fn, optimizer, device):
    """
    A function to train the model for one epoch and update weights.
    """

    # Variables to store results
    all_preds = []
    all_labels = []
    train_loss = 0

    # Iterate over batches in one epoch
    for _, batch in enumerate(tqdm(train_loader)):
        # Send data to device
        batch.to(device)

        # Put model in train mode
        model.train()

        # 1. Forward pass
        y_pred = model(batch.x.float(),
                       batch.edge_attr.float(),
                       batch.edge_index,
                       batch.batch)
    
        # 2. Calculate loss and generate predictions
        loss = torch.sqrt(loss_fn(y_pred, batch.y))
        train_loss += loss
        all_preds.append(y_pred.argmax(dim=1).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())   

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    # Calculate loss and accuracy print to screen
    train_loss /= len(train_loader)
    print(f"Train loss: {train_loss:.5f}")
    calculate_metrics(y_pred=all_preds, y_true=all_labels, context="Train")

    return train_loss


def test_step(model, test_loader, loss_fn, device):

    # Variables to store results
    all_preds = []
    all_labels = []
    test_loss = 0

    model.to(device)
    model.eval() # put model in eval mode

    # Turn on inference context manager
    with torch.inference_mode(): 
        for batch in tqdm(test_loader):
            # Send test data to device
            batch.to(device)
            
            # 1. Forward pass
            test_pred = model(batch.x.float(),
                       batch.edge_attr.float(),
                       batch.edge_index,
                       batch.batch)
            
            # 2. Calculate loss and generate predictions
            loss = torch.sqrt(loss_fn(test_pred, batch.y))
            test_loss += loss
            all_preds.append(test_pred.argmax(dim=1).cpu().detach().numpy())
            all_labels.append(batch.y.cpu().detach().numpy())

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        # Calculate loss and accuracy print to screen
        test_loss /= len(test_loader)
        print(f"Test loss: {test_loss:.5f}")
        calculate_metrics(y_pred=all_preds, y_true=all_labels, context="Test")

    return test_loss

train_losses = []
test_losses = []

# Training loop
print("Starting training loop")
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")

    # One train step
    train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
    train_losses.append(train_loss.detach().cpu().numpy())

    # One test step
    test_loss = test_step(model, test_loader, loss_fn, device)
    test_losses.append(test_loss.detach().cpu().numpy())

    scheduler.step() # LR decay

print("Training complete.")

print("Saving moodel")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

print("Plotting loss curves")
plot_loss_curve(train_losses, test_losses, output_path="loss_curve.png")