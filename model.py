import theano
import theano.tensor as T
import numpy as np
import lstm
import feedforward
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def random_init(*dimensions):
	return 2 * (np.random.rand(*dimensions) - 0.5)

def zeros_init(*dimensions):
	return np.zeros(dimensions,dtype=np.float32)



def build_stmt_encoder(P,input_size,hidden_size):
	lstm_layer = lstm.build(P,"stmt",input_size,hidden_size)
	def encode_stmt(X):
		cells,hiddens = lstm_layer(X)
		return cells[-1],hiddens[-1]
	return encode_stmt

def build_diag_encoder(P,stmt_size,hidden_size,encode_stmt):
	lstm_layer = lstm.build(P,"diag",stmt_size,hidden_size)
	def encode_diag(X,idxs):
		cells,hiddens = lstm_layer(
			T.arange(idxs.shape[0]-1),
			row_transform = (lambda i:encode_stmt(X[idxs[i]:idxs[i+1]])[0])
		)
		return cells,hiddens
	return encode_diag

def build_lookup(P,lookup_vec_size,key_vec_size,hidden_size):
	P.W_lookup_hidden_1 = 0.1 * random_init(lookup_vec_size,hidden_size)
	P.W_lookup_hidden_2 = 0.1 * random_init(key_vec_size,hidden_size)
	P.b_lookup_hidden = zeros_init(hidden_size)

	W_output = 0.1 * zeros_init(hidden_size + 1)
	W_output[-1] = -2
	P.W_lookup_output = W_output

	W = P.W_lookup_output[:-1]
	b = P.W_lookup_output[-1]

	def lookup_prep(data):
		_hiddens = T.dot(data,P.W_lookup_hidden_1) + P.b_lookup_hidden
		time = T.arange(data.shape[0])
		time_weight = T.exp(time * b)
		def lookup(key,prev_attn):
			hiddens = T.tanh(T.dot(key,P.W_lookup_hidden_2) + _hiddens)
			score = (1.-prev_attn) * T.exp(T.dot(hiddens,W))
	#		output  = gated_probs(matches)			
			norm_score = score / T.sum(score)
			time_weighted_score = time_weight * norm_score 
			output = time_weighted_score / T.sum(time_weighted_score)
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
	vocab_vectors  = 0. * random_init(vocab_size,word_rep_size)
	entity_vectors = 0. * random_init(output_size,word_rep_size)
	
	P.vocab    = vocab_vectors
	P.entities = entity_vectors
	P.W_entity_transform = 0.1 * random_init(word_rep_size,word_rep_size)
	P.b_entity_transform = zeros_init(word_rep_size)

	entity_rep = T.dot(P.entities,P.W_entity_transform) + P.b_entity_transform
	V = T.concatenate([P.vocab,entity_rep])

	encode_stmt = build_stmt_encoder(P,word_rep_size,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,stmt_hidden_size,diag_hidden_size,encode_stmt)
#	encode_qstn = build_qstn_encoder(P,word_rep_size,diag_hidden_size)
	
	P.qn2keys_cell_init = zeros_init(diag_hidden_size)
	P.qn2keys_hidden_init = zeros_init(diag_hidden_size) 
	qn2keys = lstm.build_step(P,"qn2keys",
				input_size  = diag_hidden_size + stmt_hidden_size,
				hidden_size = diag_hidden_size
			)

	lookup_prep = build_lookup(P,
			lookup_vec_size = diag_hidden_size,
			key_vec_size = stmt_hidden_size,
			hidden_size = diag_hidden_size
		)


	diag2output = feedforward.build(P,"diag2output",
				input_sizes  = [diag_hidden_size],
				hidden_sizes = [map_fun_size],
				output_size  = output_size
			)


	def qa(story,idxs,qstn):
		word_feats    = V[story]
		qn_word_feats = V[qstn]

		cum_stmt_vectors,lookup_vectors = encode_diag(word_feats,idxs)
		qn_cell,qn_hidden = encode_stmt(qn_word_feats)
		
		lookup = lookup_prep(lookup_vectors)

		attention = [None] * evidence_count
		evidence  = [None] * evidence_count

		prev_cell  = P.qn2keys_cell_init
		prev_hid   = T.tanh(P.qn2keys_hidden_init)
		input_vec = T.concatenate([cum_stmt_vectors[-1],qn_cell])
		prev_attn = 0
		for i in xrange(evidence_count): 
			prev_cell, prev_hid = qn2keys(input_vec,prev_cell,prev_hid)
			attention[i] = lookup(prev_hid,prev_attn)
			attention[i].name = "attention_%d"%i
			evidence[i] = T.dot(attention[i],cum_stmt_vectors)
			input_vec = T.concatenate([evidence[i],qn_cell])
			
			prev_attn = prev_attn + attention[i]

		final_cell, final_hidden = qn2keys(input_vec,prev_cell,prev_hid)
		output = U.vector_softmax(diag2output([final_cell]))
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

	

