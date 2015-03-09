import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

def build_stmt_key(P,input_size,output_size):
	P.W_stmt_key = np.zeros((input_size,output_size),dtype=np.float32)
	P.b_stmt_key = np.zeros((output_size,),dtype=np.float32)
	def stmt_key(stmt_vector):
		key = T.dot(stmt_vector,P.W_stmt_key) + P.b_stmt_key
		return key
	return stmt_key


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


def build(P,stmt_hidden_size,diag_hidden_size,vocab_size,output_size):
	P.W_vocab = (0.1 * 2) * (np.random.rand(len(vocab_in),10) - 0.5)
	encode_stmt = build_stmt_encoder(P,stmt_hidden_size)
	encode_diag = build_diag_encoder(P,diag_hidden_size,stmt_hidden_size,encode_stmt)
	stmt_key = build_stmt_key(P,stmt_hidden_size,diag_hidden_size)
	build_predict_subject(P,diag_hidden_size,output_size)
	def qa(story,idxs,qstn):
		word_feats = P.W_vocab[story]
		qn_word_feats = P.W_vocab[qstn]
		cum_stmt_vectors = encode_diag(word_feats,idxs)
		qn_vector = encode_stmt(qn_word_feats)
		query_vector = stmt_key(qn_vector)
		
		stmt_attention = U.vector_softmax(T.dot(query_vector,cum_stmt_vectors.T))
		selected_vector = T.dot(stmt_attention,cum_stmt_vectors)

		return stmt_attention

	return qa

if __name__ == "__main__":
	import vocab
	import data_io
	import sys
	vocab_in,vocab_out = vocab.load("qa1_vocab.pkl")
	P = Parameters()
	story = T.ivector('story')
	idxs  = T.ivector('idxs')
	qstn  = T.ivector('qstn')
	ans_evd = T.iscalar('ans_evd')
	attention = build(P,10,20,len(vocab_in),len(vocab_out))

	output = attention(story,idxs,qstn)
	params = P.values()

	cost = -T.log(output[ans_evd])
	grads = T.grad(cost,wrt=params)
	
	f = theano.function(
			inputs=[story,idxs,qstn,ans_evd],
			outputs = output,
			updates = [ (p,p-0.1*g) for p,g in zip(params,grads) ]
		)

	group_answers = data_io.group_answers(sys.argv[1])
	training_data = data_io.story_question_answer_idx(group_answers,vocab_in,vocab_out)

	for input_data,idxs,question_data,ans_w,ans_evd in training_data:
		print f(input_data,idxs,question_data,ans_evd)
