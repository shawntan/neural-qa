import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def build_stmt_encoder(P,hidden_size):
	P.W_stmt_hidden = (0.1 * 2) * (np.random.rand(hidden_size,hidden_size) - 0.5)
	P.b_stmt_hidden = np.zeros((hidden_size,),dtype=np.float32)
	P.init_stmt_hidden = np.zeros((hidden_size,),dtype=np.float32)

	def encode_stmt(X):
		def step(x,prev_hid):
			return T.nnet.sigmoid(
					x +\
					T.dot(prev_hid,P.W_stmt_hidden) +\
					P.b_stmt_hidden
				)

		hidden,_ = theano.scan(
			step,
			sequences    = [X],
			outputs_info = [T.nnet.sigmoid(P.init_stmt_hidden)],
		)
		return hidden[-1]
	return encode_stmt

def build_diag_encoder(P,hidden_size,stmt_size,encode_stmt):
	P.W_diag_stmt   = (0.1 * 2) * (np.random.rand(stmt_size,hidden_size) - 0.5)
	P.W_diag_hidden = (0.1 * 2) * (np.random.rand(hidden_size,hidden_size) - 0.5)
	P.b_diag_hidden = np.zeros((hidden_size,),dtype=np.float32)
	P.init_diag_hidden = np.zeros((hidden_size,),dtype=np.float32)

	def encode_diag(X,idxs):
		def step(i,prev_hid):
			stmt = encode_stmt(X[idxs[i]:idxs[i+1]])
			return T.nnet.sigmoid(
					T.dot(stmt,    P.W_diag_stmt) + \
					T.dot(prev_hid,P.W_diag_hidden) + \
					P.b_diag_hidden
				)
		hidden,_ = theano.scan(
			step,
			sequences    = [T.arange(idxs.shape[0]-1)],
			outputs_info = [T.nnet.sigmoid(P.init_diag_hidden)],
		)
		return hidden
	return encode_diag

def build_qstn_encoder(P,hidden_size,stmt_size,encode_stmt):
	P.W_qstn_hidden = (0.1 * 2) * (np.random.rand(hidden_size,hidden_size) - 0.5)
	P.b_qstn_hidden = np.zeros((hidden_size,),dtype=np.float32)
	P.init_qstn_hidden = np.zeros((hidden_size,),dtype=np.float32)

	def encode_qstn(X):
		def step(x,prev_hid):
			return T.nnet.sigmoid(
					x +\
					T.dot(prev_hid,P.W_qstn_hidden) +\
					P.b_qstn_hidden
				)

		hidden,_ = theano.scan(
			step,
			sequences    = [X],
			outputs_info = [T.nnet.sigmoid(P.init_qstn_hidden)],
		)
		return hidden[-1]
	return encode_qstn




def build(P,stmt_hidden_size,diag_hidden_size,vocab_size,output_size):
	encode_stmt = build_stmt_encoder(P,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,diag_hidden_size,stmt_hidden_size,encode_stmt)
	encode_qstn = build_stmt_encoder(P,stmt_hidden_size)


	return encode_diag

if __name__ == "__main__":
	import vocab
	import data_io
	import sys
	vocab_in,vocab_out = vocab.load("qa1_vocab.pkl")
	P = Parameters()
	X = T.ivector('X')
	idxs = T.ivector('idxs')
	
	P.W_vocab = (0.1 * 2) * (np.random.rand(len(vocab_in),10) - 0.5)

	encode_diag = build(P,10,20,len(vocab_in),len(vocab_out))

	output = encode_diag(P.W_vocab[X],idxs)
	params = P.values()
	cost = T.sum(output**2)
	grads = T.grad(cost,wrt=params)
	
	f = theano.function(
			inputs=[X,idxs],
			outputs = output,
			updates = [ (p,p-0.1*g) for p,g in zip(params,grads) ]
		)


	group_answers = data_io.group_answers(sys.argv[1])
	training_data = data_io.story_question_answer_idx(group_answers,vocab_in,vocab_out)

	for input_data,idxs,question_data,ans_w,ans_evd in training_data:
		print f(input_data,idxs)
		break

