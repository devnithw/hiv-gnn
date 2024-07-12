import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from rdkit import Chem
import deepchem as dc

import torch
from torch_geometric.data import Dataset


class HIVDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        '''Accepts root folder as input. No other parameters are passed since no transforms are applied to data'''
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Check if the following file(s) exits in the raw_dir directory. If not existing start download."""
        return self.filename

    @property
    def processed_file_names(self):
        """Check if the following file(s) exits in the processed_dir directory. If not existing start processing."""
        # Read CSV in raw data and get their indexes
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        self.data.index = self.data["index"]

        return [f'data_{index}.pt' for index in self.data.index]

    def download(self):
        """Download function to start downloading the raw data. Not implemented."""
        raise NotImplementedError

    def process(self):
        """Convert the SMILES data into molecule data and store as .pt files in processed folder"""

        # Read raw CSV file as Pandas dataframe | self.raw_paths[0] means the /data/raw folder
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        self.data.index = self.data["index"]

        # Initialize DeepChem featurizer
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        # Iterate through rows of the csv file
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            # Generate features using DeepChem featurizers
            out = featurizer.featurize(row["smiles"])

            # Convert to PyG graph data
            data = out[0].to_pyg_graph()

            # Get labels | utility function defined below
            data.y = self._get_label(row["HIV_active"])

            # Also attach smiles string to data object
            data.smiles = row["smiles"]

            # Save data object as .pt file
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        # Return dataset size
        return self.data.shape[0]

    def get(self, idx):
        # Load data object from .pt file according to index
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def _get_label(self, label):
        """
        Returns the labels of the molecule as a tensor
        """
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)