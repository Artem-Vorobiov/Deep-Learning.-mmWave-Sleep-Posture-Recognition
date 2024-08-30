import numpy as np
import torch

def get_dataset_statistics(dataloader):
    mean_ = 0.
    std_ = 0.
    num_samples = 0
    min_ = np.inf
    max_ = -np.inf
    for data, _, _ in dataloader:
        min_ = np.min([data.min(), min_])
        max_ = np.max([data.max(), max_])
        mean_ += data.mean().numpy()
        std_ += data.std().numpy()
        num_samples += data.size(0)

    return min_, max_, mean_ / num_samples, std_ / num_samples

def custom_collate_fn(batch):
    heatmaps = np.array([item[0] for item in batch])
    windows = [item[1] for item in batch]
    targets = np.array([item[2] for item in batch])

    return torch.as_tensor(heatmaps), windows, torch.as_tensor(targets)

class Normalize:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)
    
class Standardize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std
