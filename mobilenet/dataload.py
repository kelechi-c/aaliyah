import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from convnext_model.utils import config, read_image

hfdataset = load_dataset(config.dataset_id, spit="train", streaming=True)


class Image_dataset(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            label = torch.tensor(label, dtype=config.dtype)

            yield image, label


dataset = Image_dataset()
train_loader = DataLoader(dataset, config.batch_size, shuffle=True)
