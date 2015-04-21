import theano
import theano.tensor as T
import numpy as np
import lstm
import feedforward
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def build_stmt_encoder(P,input_size,hidden_size):
	lstm_layer = lstm.build(P,"stmt",input_size,hidden_size)
	def encode_stmt(X):
		cells,_ = lstm_layer(X)
		return cells[-1]
	return encode_stmt

def build_diag_encoder(P,stmt_size,hidden_size,encode_stmt):
	lstm_layer = lstm.build(P,"diag",stmt_size,hidden_size)
	def encode_diag(X,idxs):
		cells,hiddens = lstm_layer(
			T.arange(idxs.shape[0]-1),
			row_transform = (lambda i:encode_stmt(X[idxs[i]:idxs[i+1]]))
		)
		return cells,hiddens
	return encode_diag


def build(P,
		word_rep_size,
		stmt_hidden_size,
		diag_hidden_size,
		vocab_size,
		output_size,
		map_fun_size,
		evidence_count
	):

	P.W_vocab = (0.1 * 2) * (np.random.rand(vocab_size,word_rep_size) - 0.5)
	P.init_qn2keys_hidden = np.zeros((stmt_hidden_size,),dtype=np.float32)
	P.W_qn2keys_output    = np.zeros((stmt_hidden_size,diag_hidden_size),dtype=np.float32)
	P.b_qn2keys_output    = np.zeros((diag_hidden_size,),dtype=np.float32)


	encode_stmt = build_stmt_encoder(P,word_rep_size,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,stmt_hidden_size,diag_hidden_size,encode_stmt)

	qn2keys = lstm.build_step(P,"qn2keys",
				input_size  = diag_hidden_size,
				hidden_size = stmt_hidden_size
			)
	diag2output = feedforward.build(P,"diag2output",
				input_sizes  = evidence_count * [diag_hidden_size],
				hidden_sizes = [map_fun_size],
				output_size  = output_size
			)




	def qa(story,idxs,qstn):
		word_feats = P.W_vocab[story]
		qn_word_feats = P.W_vocab[qstn]
		cum_stmt_vectors, lookup_vectors = encode_diag(word_feats,idxs)
		qn_vector = encode_stmt(qn_word_feats)
		lookup_vectors = lookup_vectors.T

		prev_cell  = qn_vector
		input_vec  = cum_stmt_vectors[-1]
		prev_hid   = T.tanh(P.init_qn2keys_hidden)
		attention = [None] * evidence_count
		evidence  = [None] * evidence_count
		for i in xrange(evidence_count): 
			prev_cell, prev_hid = qn2keys(input_vec,prev_cell,prev_hid)
			key = T.dot(prev_hid,P.W_qn2keys_output) + P.b_qn2keys_output
			attention[i] = U.vector_softmax(T.dot(key,lookup_vectors))
			evidence[i]  = input_vec = T.dot(attention[i],cum_stmt_vectors)
			attention[i].name = "attention_%d"%i
		output = U.vector_softmax(diag2output(evidence))
		return attention,output
	return qa


