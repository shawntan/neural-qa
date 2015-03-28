import theano
import theano.tensor as T
import numpy as np
import lstm
import cPickle as pickle
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
import cPickle as pickle

def build_hidden_feedforward(P,name,input_size,hidden_size,output_size,input_count=1,output_count=1):
	for i in xrange(input_count):
		P["W_%s_hidden_%d"%(name,i)] = (0.1 * 2) * (np.random.rand(input_size,hidden_size) - 0.5)

	P["b_%s_hidden"%name] = np.zeros((hidden_size,),dtype=np.float32)
	
	for i in xrange(output_count):
		P["W_%s_output_%d"%(name,i)] = np.zeros((hidden_size,output_size),dtype=np.float32)
		P["b_%s_output_%d"%(name,i)] = np.zeros((output_size,),dtype=np.float32)

	b_hidden = P["b_%s_hidden"%name]
	def feedforward(X):
		hidden = T.tanh(
				sum(T.dot(X[i],P["W_%s_hidden_%d"%(name,i)]) for i in xrange(input_count)) +\
				b_hidden
			)
		keys = [ T.dot(hidden,P["W_%s_output_%d"%(name,i)]) + P["b_%s_output_%d"%(name,i)]
					for i in xrange(output_count) ]
		
		return keys
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
		map_fun_size,
		evidence_count
	):
	P.W_vocab = (0.1 * 2) * (np.random.rand(vocab_size,word_rep_size) - 0.5)
	encode_stmt = build_stmt_encoder(P,word_rep_size,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,stmt_hidden_size,diag_hidden_size,encode_stmt)
	stmt2key    = build_hidden_feedforward(P,"stmt2key",stmt_hidden_size,map_fun_size,diag_hidden_size,output_count=evidence_count)
	diag2output = build_hidden_feedforward(P,"stmt2output",diag_hidden_size,map_fun_size,output_size,input_count=evidence_count)
	def qa(story,idxs,qstn):
		word_feats = P.W_vocab[story]
		qn_word_feats = P.W_vocab[qstn]
		cum_stmt_vectors, lookup_vectors = encode_diag(word_feats,idxs)
		qn_vector = encode_stmt(qn_word_feats)
		keys = stmt2key([qn_vector])
		
		lookup_vectors = lookup_vectors.T
		stmt_attentions_lin = [ T.dot(key,lookup_vectors) for key in keys ]
		stmt_attentions = [
				U.vector_softmax(
					stmt_attentions_lin[i] - \
				sum(stmt_attentions_lin[:i]) - \
				sum(stmt_attentions_lin[i+1:])
				) for i in xrange(len(keys)) ]

		selected_vectors = [ T.dot(att,cum_stmt_vectors) for att in stmt_attentions ]
		output = U.vector_softmax(diag2output(selected_vectors)[0])
		return stmt_attentions,output
	return qa


