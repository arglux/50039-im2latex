import pandas as pd
import torch, cv2
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from model.models import device


class Vocabulary:
    def __init__(self, path="./meta/data_props.pkl", delimiter=" "):
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
        meta_path="./meta",
        max_length=30,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    ):
        assert group in ("test", "train")

        self.group = group
        self.vocabulary = vocabulary
        self.img_path = Path(img_path)
        self.transform = transform
        self.max_length = max_length
        self.data = pd.read_pickle(Path(meta_path) / f"df_{self.group}.pkl")
        self.data = self.data[
            (self.data.height >= 7) & (self.data.width >= 7) & (self.data.seq_len <= self.max_length)
        ].reset_index(drop=True)

    def _get_image(self, filename):
        img = 255 - cv2.imread(str(self.img_path / filename), cv2.IMREAD_GRAYSCALE)
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


def get_padding(image, max_w, max_h):
    _, height, width = imsize = image.shape
    h_padding = (max_w - width) / 2
    v_padding = (max_h - height) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    padding = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]
    return padding


def default_collate(batch):
    max_h = max([item[0].shape[1] for item in batch])
    max_w = max([item[0].shape[2] for item in batch])

    padded_ims = torch.stack(
        [
            transforms.functional.pad(item[0], get_padding(item[0], max_w, max_h))
            for item in batch
        ]
    )

    padded_targets = pad_sequence([item[1] for item in batch], batch_first=True)

    return padded_ims, padded_targets
