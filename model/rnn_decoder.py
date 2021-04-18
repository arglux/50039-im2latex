import torch
import torch.nn.functional as F

from torch import nn

class RNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, dropout_p=0.1, num_layers=1):
		super(RNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.dropout = nn.Dropout(dropout_p)

		self.clf = nn.Sequential(
			nn.Linear(self.hidden_size, self.output_size),
			nn.ReLU(),
			nn.LogSoftmax(dim=1)
		)
	def init_decoder(latent_rep):
		h =

	def step_decode(self, latent_rep, target):
		e = self.embedding(target).view(1, 1, -1)
		return e

	def forward(self, latent_rep, target):
		return self.step_decode(latent_rep, target)

if __name__ == "__main__":
	print('running rnn_decoder.py')

	torch.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rnn_decoder = RNN(hidden_size=5, output_size=5).to(device)
	tensor = torch.rand(size=(8, 10)).to(device)
	target = torch.randint(low=0, high=5, size=(8, 10)).to(device)
	logit = rnn_decoder.forward(tensor, target)

	print(rnn_decoder)
	print(tensor)
	print(f'''
	      out channel_size: {logit},
	      out img_size: {logit.shape}
	      ''')

	print('done.')
