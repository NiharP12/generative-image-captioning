"""
Dataset class for Flickr8k Image Captioning.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_dict, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.data = []
        for img_name, caps in captions_dict.items():
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                for cap in caps:
                    self.data.append((img_name, cap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = [self.vocab.word2idx["<start>"]]
        tokens.extend(self.vocab.numericalize(caption))
        tokens.append(self.vocab.word2idx["<end>"])
        return image, torch.tensor(tokens, dtype=torch.long)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class CaptionCollate:
    """Pads captions to the same length in a batch."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = [len(c) for c in captions]
        max_len = max(lengths)
        padded = torch.full((len(captions), max_len), self.pad_idx, dtype=torch.long)
        for i, cap in enumerate(captions):
            padded[i, :len(cap)] = cap
        return images, padded
