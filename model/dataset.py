import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from models import device


class Vocabulary:
    def __init__(self, path="./training_56/data_props.pkl", delimiter=" "):
        props = pd.read_pickle(path)
        self.delimiter = delimiter if delimiter else ""

        self.word2index = props.get("word2id")
        self.index2word = props.get("id2word")
        self.EOS = props.get("NullTokenID")
        self.SOS = props.get("StartTokenID")
        self.n_words = props.get("K")

    def tokenize(self, sentence):
        if self.delimiter:
            return sentence.split(self.delimiter)
        else:
            return list(sentence)

    def detokenize(self, sentence):
        return self.delimiter.join(sentence)

    def encode(self, sentence, text=False):
        indices = (
            [self.word2index[word] for word in self.tokenize(sentence)]
            if text
            else sentence
        )
        return torch.tensor(indices + [self.EOS], dtype=torch.long, device=device)

    def decode(self, indices):
        sentence = [self.index2word[ind] for ind in indices]
        return self.detokenize(sentence)

    def __len__(self):
        return self.n_words


class LatexDataset(Dataset):
    def __init__(
        self,
        vocabulary,
        group,
        img_path="./formula_images",
        meta_path="./training_56",
        transform=transforms.ToTensor(),
    ):
        assert group in ("test", "train")

        self.group = group
        self.vocabulary = vocabulary
        self.img_path = Path(img_path)
        self.transform = transform
        self.data = pd.read_pickle(Path(meta_path) / f"df_{self.group}.pkl")

    def _get_image(self, filename):
        img = Image.open(self.img_path / filename)
        if self.transform:
            img = self.transform(img)
        return img.to(device)

    def _get_target(self, target):
        return self.vocabulary.encode(target)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image = self._get_image(item.image)
        target = self._get_target(item.word2id)
        return image, target

    def __len__(self):
        return len(self.data)
