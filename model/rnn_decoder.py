import torch
import torch.nn.functional as F

from torch import nn

class RNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, latent_rep_channel, dropout_p=0.1):
		super(RNN, self).__init__()
		self.lstm = nn.LSTMCell(latent_rep_channel + embed_size, embed_size)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		# self.dropout = nn.Dropout(dropout_p)
		self.clf = nn.Sequential(
			nn.Linear(self.hidden_size, self.output_size),
			nn.ReLU(),
			nn.LogSoftmax(dim=1)
		)

		# lstm cell weight definitions
		self.W_h = nn.Linear(latent_rep_channel, hidden_size)
		self.W_c = nn.Linear(latent_rep_channel, hidden_size)
		self.W_o = nn.Linear(latent_rep_channel, hidden_size)

	def init_lstm(latent_rep):
		"""
		initialize lstm at t=0 using the mean of latent_rep [B, H*W, C=128]
		"""
		mean_latent_rep = latent_rep.mean(dim=1)
		h_0 = torch.tanh(self.W_h(mean_latent_rep)) # [B, hidden_size]
		c_0 = torch.tanh(self.W_c(mean_latent_rep))
		o_0 = torch.tanh(self.W_o(mean_latent_rep))
		return (h_0, c_0), o_0

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
