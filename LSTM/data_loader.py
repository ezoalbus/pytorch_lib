import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

from prepare_data import prepare_data
from preprocess import Text2IDseq


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self):
        text2idseq = Text2IDseq()
        self.data, self.labels = text2idseq.make_dataset_idx()

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        return data, labels


def split_dataset(dataset, split_ratio=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=split_ratio)
    split_dataset = {}
    split_dataset['train'] = Subset(dataset, train_idx)
    split_dataset['test'] = Subset(dataset, test_idx)
    return split_dataset


def get_dataloader(batch_size=100):
    dataset = Mydatasets()
    dataset = split_dataset(dataset)
    dataloaders = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True) \
                     for x in ['train','test']}

    return dataloaders