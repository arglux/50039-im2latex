import torch
import torch.nn.functional as F

from torch import nn

class RNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, latent_rep_channel, dropout_p=0.1):
		super(RNN, self).__init__()
		self.lstm = nn.LSTMCell(hidden_size + embed_size, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.dropout = nn.Dropout(dropout_p)

		# lstm cell weight definitions
		self.W_h = nn.Linear(latent_rep_channel, hidden_size)
		self.W_c = nn.Linear(latent_rep_channel, hidden_size)
		self.W_o = nn.Linear(latent_rep_channel, hidden_size)

	def init_lstm(latent_rep):
		"""
		initialize lstm at t=0 using the mean of latent_rep [B, H*W, C=128]
		"""
		mean_latent_rep = latent_rep.mean(dim=1)
		h_0 = torch.tanh(self.W_h(mean_latent_rep)) # [B, h]
		c_0 = torch.tanh(self.W_c(mean_latent_rep))
		o_0 = torch.tanh(self.W_o(mean_latent_rep))
		return (h_0, c_0), o_0

	def step_decode(self, latent_rep, target, lstm_states):
		"""
		calculate lstm state at t from t-1 state
		"""
		(h_t, c_t), o_t = lstm_states
		embedding = self.embedding(target) # [B, e]
		inp = torch.cat([o_t, hidden_size], dim=1) # [B, h+e]
		(h_t, c_t) = self.lstm(inp, (h_t, c_t))
		h_t = self.dropout(h_t)
		c_t = self.dropout(c_t)

		# apply attention
		# context_t, attn_scores = self._get_attn(enc_out, h_t)

	def decode(self, latent_rep, formulas):
		(h_t, c_t), o_t = self.init_lstm(latent_rep)
		max_len = formulas.size(1)
		logits = []

		for t in range(max_len):
			target = formulas[:, t:t+1] # get 1 char
		return self.step_decode(latent_rep, formula)

if __name__ == "__main__":
	print('running rnn_decoder.py')

	torch.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rnn_decoder = RNN(vocab_size=10, embed_size=5, hidden_size=7, latent_rep_channel=8).to(device)
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
