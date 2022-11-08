import os

import numpy as np

import torch 
from torch.utils.data import TensorDataset, DataLoader


"""data"""
def load_data_ili(data_folder="data/ILI", training_steps=10, test_part=2,
                 batch_size=64):
    path_0 = os.path.join(data_folder, "ili_part0.npy")
    path_1 = os.path.join(data_folder, "ili_part1.npy")
    path_2 = os.path.join(data_folder, "ili_part2.npy")
    with open(path_0, 'rb') as f:
        part0 = np.load(f)    
    with open(path_1, 'rb') as f:
        part1 = np.load(f)    
    with open(path_2, 'rb') as f:
        part2 = np.load(f)
    if test_part==0:
        test_data = part0
        training_data = np.concatenate([part2,part1], axis=0)
    if test_part==1:
        test_data = part1
        training_data = np.concatenate([part0, part2], axis=0)
    if test_part==2:
        test_data = part2
        training_data = np.concatenate([part0,part1], axis=0)

    valid_data = training_data[-int(len(training_data)/4):]
    training_data = training_data[:-int(len(training_data)/4)]

    training_set = TensorDataset(torch.from_numpy(training_data[:,:training_steps]).float().unsqueeze(-1), 
                             torch.from_numpy(training_data[:,training_steps:]).float().unsqueeze(-1))
    valid_set = TensorDataset(torch.from_numpy(valid_data[:,:training_steps]).float().unsqueeze(-1), 
                             torch.from_numpy(valid_data[:,training_steps:]).float().unsqueeze(-1))
    test_set = TensorDataset(torch.from_numpy(test_data[:,:training_steps]).float().unsqueeze(-1),
                             torch.from_numpy(test_data[:,training_steps:]).float().unsqueeze(-1))
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return training_loader, valid_loader, test_loader


