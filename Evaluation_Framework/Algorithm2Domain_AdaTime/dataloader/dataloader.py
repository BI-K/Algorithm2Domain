import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

#from ..dataloader.data_augmenter import augment_dataset
import os, sys
import numpy as np
import random


class SafeNormalize:
    def __init__(self, mean, std):
            self.mean = mean
            self.std = std
            # Identify channels with zero variance
            self.zero_var_mask = std < 1e-5
            
    def __call__(self, x):
            # Reshape for channel-wise normalization
            original_shape = x.shape
            x_reshaped = x.reshape(self.mean.shape[0], -1)
            
            # Apply normalization only to channels with variance
            normalized = torch.zeros_like(x_reshaped)
            
            # For channels with variance, apply standard normalization
            valid_channels = ~self.zero_var_mask
            if valid_channels.any():
                normalized[valid_channels] = (x_reshaped[valid_channels] - self.mean[valid_channels].unsqueeze(1)) / self.std[valid_channels].unsqueeze(1)
            
            # For channels with zero variance, set to 0 (they're already 0 in the initialized tensor)
            # This explicitly handles the zero variance case
            normalized[self.zero_var_mask] = 0.0
            
            return normalized.reshape(original_shape)





class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs, dtype, is_source):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset["labels"]
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.tensor(y_data, dtype=torch.float64)


        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.tensor(x_data, dtype=torch.float64)

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        #if dtype == "train":
        #    x_data, y_data = augment_dataset(self, dataset_configs, x_data, y_data, is_source=is_source)


        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            # replace all values of data_std that are too small by 1e+5
            safe_std = torch.where(data_std < 1e-5, torch.tensor(1.0, device=data_std.device), data_std)
            self.transform = transforms.Normalize(mean=data_mean, std=safe_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len

class Load_MergedDataset(Load_Dataset):
    def __init__(self, datasets, dataset_configs, dtype, is_source):
        # Merge datasets
        merged_samples = torch.cat([torch.tensor(dataset["samples"]) if isinstance(dataset["samples"], np.ndarray) else dataset["samples"] for dataset in datasets], dim=0)
        merged_labels = torch.cat([torch.tensor(dataset["labels"]) if isinstance(dataset["labels"], np.ndarray) else dataset["labels"] for dataset in datasets], dim=0) if datasets[0]["labels"] is not None else None
        
        merged_dataset = {
            "samples": merged_samples,
            "labels": merged_labels
        }
        
        super().__init__(merged_dataset, dataset_configs, dtype, is_source)




def data_generator(data_path, domain_id, dataset_configs, hparams, dtype, is_source):

    # Loading datasets
    if dtype == "all":
        dataset_file_train = torch.load(os.path.join(data_path, f"train_{domain_id}.pt"), weights_only=False)
        dataset_file_test = torch.load(os.path.join(data_path, f"test_{domain_id}.pt"), weights_only=False)
        dataset_files = [dataset_file_train, dataset_file_test]
        dataset = Load_MergedDataset(dataset_files, dataset_configs, dtype=dtype, is_source=is_source)
    else:
        dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"), weights_only=False)
        dataset = Load_Dataset(dataset_file, dataset_configs, dtype=dtype, is_source=is_source)

    if dtype == "test" or dtype == "all":  # you don't need to shuffle or drop last batch while testing
        shuffle  = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=hparams["batch_size"],
                                                shuffle=shuffle, 
                                                drop_last=drop_last, 
                                                num_workers=0)

    return data_loader




def few_shot_data_generator(data_loader, dataset_configs, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    few_shot_dataset = Load_Dataset(few_shot_dataset, dataset_configs, dtype="test", is_source=True)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader

