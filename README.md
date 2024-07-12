# HIV Inhibitor Classification using GNN
A Graph Neural Network with Graph Attention Layers paired with TopK pooling to classify and generate HIV inhibitor molecules

## Overview

## Data
The data for inhibitor molecules was obtained from [MoleculeNet](https://moleculenet.org/datasets-1) data repositoty. The file `HIV.scv` inclues experimentally measured abilities to inhibit HIV replication. This CSV includes three fields representing `molecule smiles` string, `activity` and `HIV_active` status. However the data is skewed in a way that there are 39684 samples for negative class (not HIV active) and 1443 samples for positive class (HIV active).

Following are some molecules visualized using RDKit Chem module.

![HIV Negative molecules](images/hiv_negative.png)

*Visualization of random HIV negative molecules*

![HIV Positive molecules](images/hiv_positive.png)

*Visualization of random HIV positive molecules*

These visualizations can be generated from the following code

```python 
from rdkit import Chem
from rdkit.Chem import Draw

# Convert SMILES to RDKit molecule object
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles]

# Draw the molecule into grid
img = Draw.MolsToGridImage(molecules, molsPerRow=3)
```

## Dataset Preprocessing

To handle class imbalance, the dataset was preprocessed before loading into a PyTorch Dataset class. Oversampling was used on the positive samples and increased them upto 10101 samples. Class imbalance will further be addressed by weighted loss during training.

The processed data was then seperated into two csv files using `sklearn.model_selection.train_test_split()` with a random split of 20:80. The csv files are stored in `data/train/raw` and `data/test/raw` seperately for proper working of the `torch_geometric.data.Dataset()` class. According to documentation, the Dataset class processes the data in the raw folder and caches in `root/processed` as a .pt file. 

To set up data for training run the `preprocess.py` file. The `dataset.py` file includes the class definition for the HIVDatset() object.

## Model

## Model Training

## Technologies used
- PyTorch Geometric
- RDKit
- DeepChem
- Google Colab

## References
