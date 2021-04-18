import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


class LatexDataset(Dataset):
	def __init__(self, group, img_path="./data/formula_images", meta_path="./data/training_56", transform=transforms.ToTensor()):
		assert group in ("test", "train")

		self.group = group
		self.img_path = Path(img_path)
		self.meta_path = Path(meta_path)
		self.transform = transform
		self.data = pd.read_pickle(self.meta_path / f"df_{self.group}.pkl")

		data_props = pd.read_pickle(self.meta_path / "data_props.pkl")
		self.id2word = data_props.get("id2word")
		self.word2id = data_props.get("word2id")

	def _get_image(self, filename):
		img = Image.open(self.img_path / filename)
		if self.transform:
		img = self.transform(img)
		return img

	def __getitem__(self, index):
		item = self.data.iloc[index]
		image = self._get_image(item.image)
		target = item.padded_seq
		return image, target

	def __len__(self):
		return len(self.data)

if __name__ == "__main__":
	print('running dataset.py')
	test_set = LatexDataset('test')
	print(len(test_set))
