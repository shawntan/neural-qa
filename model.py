import theano
import theano.tensor as T
import numpy as np
import lstm
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def build_hidden_feedforward(P,name,input_size,hidden_size,output_size):
	P["W_%s_hidden"%name] = (0.1 * 2) * (np.random.rand(input_size,hidden_size) - 0.5)
	P["b_%s_hidden"%name] = np.zeros((hidden_size,),dtype=np.float32)
	P["W_%s_output"%name] = np.zeros((hidden_size,output_size),dtype=np.float32)
	P["b_%s_output"%name] = np.zeros((output_size,),dtype=np.float32)
	W_hidden = P["W_%s_hidden"%name]
	b_hidden = P["b_%s_hidden"%name]
	W_output = P["W_%s_output"%name]
	b_output = P["b_%s_output"%name]
	def feedforward(X):
		hidden = T.tanh(T.dot(X,W_hidden) + b_hidden)
		key = T.dot(hidden,W_output) + b_output 
		return key
	return feedforward


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
		map_fun_size
	):
	P.W_vocab = (0.1 * 2) * (np.random.rand(vocab_size,word_rep_size) - 0.5)
	encode_stmt = build_stmt_encoder(P,word_rep_size,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,stmt_hidden_size,diag_hidden_size,encode_stmt)
	stmt2key    = build_hidden_feedforward(P,"stmt2key",stmt_hidden_size,map_fun_size,diag_hidden_size)
	diag2output = build_hidden_feedforward(P,"stmt2output",diag_hidden_size,map_fun_size,output_size)
	def qa(story,idxs,qstn):
		word_feats = P.W_vocab[story]
		qn_word_feats = P.W_vocab[qstn]
		cum_stmt_vectors, lookup_vectors = encode_diag(word_feats,idxs)
		qn_vector = encode_stmt(qn_word_feats)
		key = stmt2key(qn_vector)
		stmt_attention = U.vector_softmax(T.dot(key,lookup_vectors.T))
		selected_vector = T.dot(stmt_attention,cum_stmt_vectors)
		output = U.vector_softmax(diag2output(selected_vector))
		return stmt_attention,output
	return qa


