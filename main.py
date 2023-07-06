import os
import pandas as pd
from skimage import io, transform
import random
import shutil

import torch
from torch.utils.data import Dataset
import os

# Rutas de las carpetas de destino
destino_entrenamiento = '/media/user_home0/sgoyesp/Proyecto/baseline/train'
destino_validacion = '/media/user_home0/sgoyesp/Proyecto/baseline/valid'
destino_prueba = '/media/user_home0/sgoyesp/Proyecto/baseline/test'

# Crear carpetas de destino si no existen
os.makedirs(destino_entrenamiento, exist_ok=True)
os.makedirs(destino_validacion, exist_ok=True)
os.makedirs(destino_prueba, exist_ok=True)

# Rutas de las carpetas de destino

# Dataset class
class ISICDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, labels_csv, data_path, transform=None):
        super(ISICDataset, self).__init__()
        'Initialization'
        self.labels = pd.read_csv(labels_csv)
        self.list_IDs = os.listdir(data_path)
        self.list_IDs = random.sample(self.list_IDs, len(self.list_IDs))
        self.transform = transform
        self.data_path = data_path
        total_samples = len(self.list_IDs)
        train_samples = int(total_samples * 0.8)
        val_samples = int(total_samples * 0.1)

        train_files = self.list_IDs[:train_samples]
        val_files = self.list_IDs[train_samples:train_samples + val_samples]
        test_files = self.list_IDs[train_samples + val_samples:]

        for file in train_files:
            shutil.copyfile(os.path.join(self.data_path, file), os.path.join(destino_entrenamiento, file))
    
        for file in val_files:
            shutil.copyfile(os.path.join(self.data_path, file), os.path.join(destino_validacion, file))

        for file in test_files:
            shutil.copyfile(os.path.join(self.data_path, file), os.path.join(destino_prueba, file))
     
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        # Load data and get label
        label = self.labels[ID]
        image = io.imread(os.path.join(self.data_path, ID))
        # image = transform.resize(image, (375,500))
        if self.transform:
            image = self.transform(image)
        return image, label


#breakpoint()
dataset = ISICDataset("/media/user_home0/sgoyesp/Proyecto/ISIC_2019_Training_Labels.csv", "/media/user_home0/sgoyesp/Proyecto/ISIC_2019_Training_Input")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

