import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        left = self.df.loc[idx, "left"]
        right = self.df.loc[idx, "right"]
        label = self.df.loc[idx, "label"]

        # Return data as a dictionary or tuple
        return {'left': left, 'right': right, 'label': label}


class LfwLoader:
    def __init__(self, df_train, df_val, df_test, experiment):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def get_train_loader(self):
        dataset = CustomDataset(self.df_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def get_val_loader(self):
        dataset = CustomDataset(self.df_val)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def get_test_loader(self):
        dataset = CustomDataset(self.df_val)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader
