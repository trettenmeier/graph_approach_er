import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class LFWDataset(Dataset):
    def __init__(self, pairs_file, root_dir, transform=None):
        self.pairs = self.load_pairs(pairs_file)
        self.root_dir = root_dir
        self.transform = transform

    def load_pairs(self, pairs_file):
        pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip the header line
            for line in lines:
                items = line.strip().split()
                if len(items) == 3:
                    # Positive pair (same person)
                    person, idx1, idx2 = items
                    pairs.append((person, int(idx1), person, int(idx2), 1))
                elif len(items) == 4:
                    # Negative pair (different persons)
                    person1, idx1, person2, idx2 = items
                    pairs.append((person1, int(idx1), person2, int(idx2), 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person1, idx1, person2, idx2, label = self.pairs[idx]

        img1_path = os.path.join(self.root_dir, person1, f"{person1}_{idx1:04d}.jpg")
        img2_path = os.path.join(self.root_dir, person2, f"{person2}_{idx2:04d}.jpg")

        image1 = Image.open(img1_path).convert("RGB")
        image2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        label = torch.tensor(label, dtype=torch.float32)

        return image1, image2, label

    def get_pairs(self):
        return self.pairs


class LfwDataProcessing:
    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # or any size you need
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
        ])

        # train and val data
        lfw_dataset = LFWDataset(
            pairs_file=os.path.join(self.path_to_data, 'pairsDevTrain.txt'),
            root_dir=os.path.join(self.path_to_data, 'lfw-deepfunneled'),
            transform=self.transform
        )

        dataloader = DataLoader(lfw_dataset, batch_size=1, shuffle=True)

        data = {
            "left": [],
            "right": [],
            "label": []
        }

        for img1, img2, label in dataloader:
            data["left"].append(img1.squeeze())
            data["right"].append(img2.squeeze())
            data["label"].append(int(label))

        df = pd.DataFrame(data)

        df_train, df_val = train_test_split(df, random_state=123, test_size=0.1)
        self.df_train = df_train.reset_index(drop=True)
        self.df_val = df_val.reset_index(drop=True)

    def get_df_train(self):
        return self.df_train.copy(deep=True)

    def get_df_val(self):
        return self.df_val.copy(deep=True)

    def get_df_test(self):
        lfw_dataset = LFWDataset(
            pairs_file=os.path.join(self.path_to_data, 'pairsDevTest.txt'),
            root_dir=os.path.join(self.path_to_data, 'lfw-deepfunneled'),
            transform=self.transform
        )

        dataloader = DataLoader(lfw_dataset, batch_size=1, shuffle=True)

        data = {
            "left": [],
            "right": [],
            "label": []
        }

        for img1, img2, label in dataloader:
            data["left"].append(img1.squeeze())
            data["right"].append(img2.squeeze())
            data["label"].append(int(label))

        return pd.DataFrame(data)
