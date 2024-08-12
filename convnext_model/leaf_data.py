import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from convnext_model.utils import config, read_image

hfdataset = load_dataset(config.dataset_id, spit="train", streaming=True)
hfdataset = hfdataset.take(config.split)


class LeafDataset(IterableDataset):
    def __init__(self, dataset=hfdataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            label = torch.tensor(label, dtype=config.dtype)

            yield image, label


img_dataset = LeafDataset()
train_loader = DataLoader(img_dataset, config.batch_size, shuffle=True)
