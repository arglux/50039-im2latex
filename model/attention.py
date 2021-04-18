import torch
import torch.nn.functional as F

from torch import nn

class RNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def encode(self, images):
		latent_rep = self.rnn(images) # [B, C=256, H, W]
		batch_size, channel_size, img_height, img_width = latent_rep.shape
		# latent_rep = add_positional_features(latent_rep)
		return latent_rep, batch_size, channel_size, (img_height, img_width)

	def forward(self, images):
		latent_rep, batch_size, channel_size, img_size = self.encode(images)
		latent_rep = latent_rep.view(batch_size, -1)
		return latent_rep, batch_size, channel_size, img_size

if __name__ == "__main__":
	print('running rnn_decoder.py')

	torch.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cnn_encoder = RNN().to(device)
	tensor = torch.rand(size=(8, 1, 20, 20)).to(device)
	latent_rep, _, channel_size, img_size = cnn_encoder.forward(tensor)

	print(cnn_encoder)
	print(latent_rep)
	print(f'''
	      out channel_size: {channel_size},
	      out img_size: {img_size}
	      ''')

	print('done.')
