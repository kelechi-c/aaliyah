import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from .utils_config import config
from ..random_utils import read_image

hfdataset = load_dataset(config.leaf_dataset_id, spit="train", streaming=True)


class Image_dataset(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"], config.image_size)
            label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            label = torch.tensor(label, dtype=config.dtype)

            yield image, label


dataset = Image_dataset(hfdataset)
train_loader = DataLoader(dataset, config.batch_size, shuffle=True)
