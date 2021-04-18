import torch
import torch.nn.functional as F

from torch import nn

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),

			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU()
		)

	def encode(self, images):
		latent_rep = self.cnn(images) # [B, C=256, H, W]
		batch_size, channel_size, img_height, img_width = latent_rep.shape
		# latent_rep = add_positional_features(latent_rep)
		return latent_rep, batch_size, channel_size, (img_height, img_width)

	def forward(self, images):
		latent_rep, batch_size, channel_size, img_size = self.encode(images)
		latent_rep = latent_rep.view(batch_size, -1)
		return latent_rep, batch_size, channel_size, img_size

if __name__ == "__main__":
	print('running cnn_encoder.py')

	torch.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cnn_encoder = CNN().to(device)
	tensor = torch.rand(size=(8, 1, 20, 20)).to(device)
	latent_rep, _, channel_size, img_size = cnn_encoder.forward(tensor)

	print(cnn_encoder)
	print(latent_rep)
	print(f'''
	      out channel_size: {channel_size},
	      out img_size: {img_size}
	      ''')

	print('done.')
