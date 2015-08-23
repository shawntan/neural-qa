import theano
import theano.tensor as T
import numpy as np
import lstm
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def random_init(*dimensions):
	#return 2 * (np.random.rand(*dimensions) - 0.5)
	return np.random.randn(*dimensions)

def zeros_init(*dimensions):
	return np.zeros(dimensions,dtype=np.float32)



def build_stmt_encoder(P,name,input_size,hidden_size):
	lstm_layer = lstm.build(P,name,input_size,hidden_size)
	def encode_stmt(X):
		cells,hiddens = lstm_layer(X)
		return cells[-1],hiddens[-1]
	return encode_stmt


def build_diag_encoder(P,stmt_size,hidden_size,output_size,encode_stmt):
#	P.W_stmt_diag_hidden = random_init(stmt_size,output_size)
#	P.W_cumstmt_diag_hidden = random_init(hidden_size,output_size)
	lstm_layer = lstm.build(P,"diag",stmt_size,hidden_size)
	def encode_diag(X,idxs):
		def encode_sentence(i):
			word_vecs = X[idxs[i]:idxs[i+1]]
			return encode_stmt(word_vecs)[0] 
		stmt_vecs,_ = theano.map(
				encode_sentence,
				sequences=[T.arange(idxs.shape[0]-1)]
			)
		cells,hiddens = lstm_layer(stmt_vecs)
#		output = T.dot(stmt_vecs,P.W_stmt_diag_hidden) +\
#				 T.dot(hiddens,P.W_cumstmt_diag_hidden)
		return cells,hiddens
	return encode_diag

def build_lookup(P,data_size,state_size,hidden_size=256):
	def init(input_size,output_size):
		return np.random.uniform(
				low  = - np.sqrt(6. / (input_size + output_size)),
				high =   np.sqrt(6. / (input_size + output_size)),
				size =  (input_size,output_size)
			)
	P.W_attention_data_hidden = init(data_size,hidden_size) #0.001 * random_init(data_size,hidden_size)
	P.W_attention_state_hidden = init(state_size,hidden_size) #0.001 * random_init(state_size,hidden_size)
	P.b_attention_hidden = np.zeros((hidden_size,))
	P.W_attention = 0.1 * random_init(hidden_size)
	def lookup_prep(data):
		hidden_contribution = T.dot(data,P.W_attention_data_hidden)
		def lookup(key,prev_attn):
			score  = T.dot(
					T.nnet.sigmoid(
						hidden_contribution + \
						T.dot(key,P.W_attention_state_hidden) + \
						P.b_attention_hidden
					),
					P.W_attention
				)
			score_max = T.max(score)
			matches = T.exp(score - score_max) #* ( 1 - prev_attn )
			match_prob = matches / (T.sum(matches))
#			match_prob = U.vector_softmax(score)
#			time_weighted_match = match_prob * time_weight
			#output = time_weighted_match / T.sum(time_weighted_match)
			output = match_prob
			return output
		return lookup
	return lookup_prep

def build(P,
		word_rep_size,
		stmt_hidden_size,
		diag_hidden_size,
		vocab_size,
		output_size,
		map_fun_size,
		evidence_count
	):

	vocab_vectors = 0.001 * random_init(vocab_size,word_rep_size)
	P.vocab = vocab_vectors
	V = P.vocab

	encode_qstn = encode_stmt = build_stmt_encoder(P,"stmt",word_rep_size,stmt_hidden_size)
	#encode_qstn = build_stmt_encoder(P,"qstn",word_rep_size,diag_hidden_size)
	encode_diag = build_diag_encoder(P,
			stmt_size   = stmt_hidden_size,
			hidden_size = diag_hidden_size,
			output_size = diag_hidden_size,
			encode_stmt = encode_stmt
		)

	qn2keys = lstm.build_step(P,"qn2keys",
				input_size  = diag_hidden_size,
				hidden_size = diag_hidden_size
			)

	lookup_prep = build_lookup(P,
			data_size = diag_hidden_size,
			state_size = diag_hidden_size
		)

#	diag2output = feedforward.build(P,"diag2output",
#				input_sizes  = [diag_hidden_size],
#				hidden_sizes = [map_fun_size],
#				output_size  = vocab_size
#			)
	P.W_output_vocab = 0.01 * random_init(diag_hidden_size,vocab_size)
	P.b_output_vocab = 0.00 * np.zeros((vocab_size,))


	def qa(story,idxs,qstn):
		word_feats    = V[story]
		qn_word_feats = V[qstn]

		diag_cells,diag_hiddens = encode_diag(word_feats,idxs)
		qn_cell,qn_hidden = encode_qstn(qn_word_feats)
		
		lookup = lookup_prep(diag_hiddens)

		attention = [None] * evidence_count
		evidence  = [None] * evidence_count


		prev_cell,prev_hidden = qn_cell,qn_hidden
		prev_attn = 0
		alpha = 0.0
		input_vec = T.mean(diag_cells,axis=0)
		for i in xrange(evidence_count): 
			prev_cell, prev_hidden = qn2keys(input_vec,prev_cell,prev_hidden)
			attention[i] = lookup(prev_hidden,prev_attn)
			attention[i].name = "attention_%d"%i
			evidence[i] = input_vec = T.sum(attention[i].dimshuffle(0,'x') * diag_cells,axis=0)
								#	alpha * T.mean(diag_vectors,axis=0)
			prev_attn = prev_attn + attention[i]
		final_cell, final_hidden = prev_cell,prev_hidden

		output = U.vector_softmax(T.dot(final_hidden,P.W_output_vocab) + P.b_output_vocab)
		return attention,output
	return qa

def gated_probs(gate_seq):
	gate_seq = T.concatenate([[np.float32(1.)],gate_seq[1:]])
	probs,_ = theano.scan(
			(lambda p,neg_prev:( p * neg_prev, (1-p)*neg_prev)),
			sequences = gate_seq,
			go_backwards = True,
			outputs_info = [None,1.],
		)
	return probs[0][::-1]


if __name__ == "__main__":
	from theano_toolkit import hinton

	data = 0.5 * np.ones(5,dtype=np.float32)
	data[-2] = 1
	print(gated_probs(data).eval())

	

